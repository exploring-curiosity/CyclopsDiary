"""Storage connectors over fsspec.

One rule: raw objects are never copied into the store. The connector reads
metadata (size, checksum, storage class), lists prefixes, serves byte
ranges, and fetches an object to a SCRATCH path only when an encoder must
see every byte, after which the scratch copy is deleted by the caller.

Tier is decided by ACCESS semantics, not price:
  hot  = readable now (S3 Standard / IA / Glacier Instant Retrieval /
         Intelligent-Tiering in its instant tiers, every GCS class, Azure
         Hot / Cool / Cold, local disk)
  cold = a restore request must complete first (S3 Glacier Flexible
         Retrieval and Deep Archive, Intelligent-Tiering objects in an
         archive access tier, Azure Archive)
The raw provider string is kept in `storage_class` for cost reporting.

Provider clients are optional extras: s3fs (S3, MinIO, R2), gcsfs (GCS),
adlfs (Azure). Local paths and memory:// need nothing.

An endpoint and its credentials are OPTIONS OF A URL PREFIX, registered
with `set_options`, never a process-wide global: one process can hold a
customer's bucket on AWS and the development mock on localhost at once,
and every read of s3://bucket/key finds its own endpoint. Environment
variables (ELIDEDB_S3_ENDPOINT, _KEY, _SECRET, _REGION) are the default
for any s3:// URL nobody registered. Addressing is path-style when an
endpoint is named, because an endpoint given by URL has no per-bucket DNS,
and whatever boto3 resolves otherwise, because that is what AWS serves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import fsspec

_OPTIONS: dict[str, dict] = {}


def s3_options(endpoint: str | None = None, key: str | None = None, secret: str | None = None,
               region: str | None = None, token: str | None = None) -> dict:
    """fsspec storage options for s3fs against one endpoint.

    Path-style addressing is set only when an endpoint is named. An
    endpoint given by URL has no per-bucket DNS, so path-style is the only
    thing that reaches it; AWS is the other way round, serving virtual-host
    addressing and deprecating path-style, which already fails outright for
    a bucket whose name contains a dot over TLS. Forcing it everywhere made
    the switch from a local endpoint to AWS not a switch.
    """
    o: dict = {}
    ck = {}
    if endpoint:
        o["config_kwargs"] = {"s3": {"addressing_style": "path"}}
        ck["endpoint_url"] = endpoint
    if region:
        ck["region_name"] = region
    if ck:
        o["client_kwargs"] = ck
    if key:
        o["key"] = key
    if secret:
        o["secret"] = secret
    if token:
        o["token"] = token
    return o


ROLE_SESSION_S = 3600


def assume_role(role_arn: str, external_id: str = "", region: str = "",
                endpoint: str | None = None, duration_s: int = ROLE_SESSION_S,
                session_name: str = "elidedb") -> dict:
    """Temporary credentials for a role a customer granted, as the key,
    secret and token `s3_options` takes.

    The AssumeRole call is signed with this deployment's own identity, and
    only the temporary triple reaches the reader: nothing of the
    customer's is stored, and the external ID they wrote into the trust
    policy is presented every time. The credentials expire, so this is
    called when a source is registered for a job rather than kept.
    """
    import boto3
    ck: dict = {}
    if region:
        ck["region_name"] = region
    if endpoint:
        ck["endpoint_url"] = endpoint
    kw = {"RoleArn": role_arn, "RoleSessionName": session_name,
          "DurationSeconds": int(duration_s)}
    if external_id:
        kw["ExternalId"] = external_id            # AWS rejects an empty one
    c = boto3.client("sts", **ck).assume_role(**kw)["Credentials"]
    return {"key": c["AccessKeyId"], "secret": c["SecretAccessKey"], "token": c["SessionToken"]}


def set_options(prefix: str, options: dict | None) -> None:
    """Register (or with None, forget) the storage options for every URL
    under `prefix`, e.g. "s3://customer-bucket/"."""
    if options is None:
        _OPTIONS.pop(prefix, None)
    else:
        _OPTIONS[prefix] = dict(options)


def _normalise(url: str) -> str:
    if "://" not in url:
        return f"file://{Path(url).expanduser().resolve()}"
    return url


def options_for(url: str) -> dict:
    """The registered options of the longest matching prefix; for an
    unregistered s3:// URL the environment's; otherwise nothing."""
    url = _normalise(url)
    best = max((p for p in _OPTIONS if url.startswith(p)), key=len, default=None)
    if best is not None:
        return dict(_OPTIONS[best])
    if url.startswith("s3://"):
        env = os.environ.get
        if any(env(k) for k in ("ELIDEDB_S3_ENDPOINT", "ELIDEDB_S3_KEY", "ELIDEDB_S3_SECRET", "ELIDEDB_S3_REGION")):
            return s3_options(endpoint=env("ELIDEDB_S3_ENDPOINT"), key=env("ELIDEDB_S3_KEY"),
                              secret=env("ELIDEDB_S3_SECRET"), region=env("ELIDEDB_S3_REGION"))
    return {}

_S3_COLD = {"GLACIER", "DEEP_ARCHIVE"}
_S3_ARCHIVED = {"ARCHIVE_ACCESS", "DEEP_ARCHIVE_ACCESS"}
_AZ_COLD = {"archive"}


@dataclass(frozen=True)
class ObjectRef:
    url: str            # canonical: scheme://path
    size: int
    etag: str           # provider checksum or "" when unknown
    storage_class: str  # provider string, "" when unknown
    tier: str           # "hot" | "cold"
    modified: str       # provider timestamp as text, "" when unknown


def parse_url(url: str, options: dict | None = None):
    """-> (scheme, filesystem, path). A bare path is a local file URL; the
    filesystem is built with the URL's registered options (see
    `set_options`) unless `options` is given."""
    url = _normalise(url)
    scheme = url.split("://", 1)[0]
    fs, path = fsspec.core.url_to_fs(url, **(options if options is not None else options_for(url)))
    return scheme, fs, path


def _canonical(scheme: str, fs, path: str) -> str:
    p = fs._strip_protocol(path) if hasattr(fs, "_strip_protocol") else path
    if p.startswith(f"{scheme}://"):
        return p
    if scheme == "file":
        return f"file://{p}"
    return f"{scheme}://{p.lstrip('/')}"


def tier_of(info: dict) -> str:
    sc = str(info.get("StorageClass") or info.get("storage_class") or "").upper()
    if sc in _S3_COLD:
        return "cold"
    if str(info.get("ArchiveStatus") or "").upper() in _S3_ARCHIVED:
        return "cold"
    az = str(info.get("access_tier") or info.get("AccessTier") or "").lower()
    if az in _AZ_COLD:
        return "cold"
    return "hot"


def _ref_from_info(scheme: str, fs, path: str, info: dict) -> ObjectRef:
    etag = (info.get("ETag") or info.get("etag") or info.get("md5Hash")
            or info.get("crc32c") or "")
    sc = (info.get("StorageClass") or info.get("storageClass")
          or info.get("access_tier") or "")
    mod = (info.get("LastModified") or info.get("updated")
           or info.get("mtime") or "")
    return ObjectRef(url=_canonical(scheme, fs, path),
                     size=int(info.get("size") or 0),
                     etag=str(etag).strip('"'), storage_class=str(sc),
                     tier=tier_of(info), modified=str(mod))


def stat(url: str) -> ObjectRef:
    scheme, fs, path = parse_url(url)
    return _ref_from_info(scheme, fs, path, fs.info(path))


def list_objects(url: str, suffixes=(), recursive: bool = True) -> list[ObjectRef]:
    """Every object under `url` whose name ends with one of `suffixes`
    (case-insensitive; empty = everything). Directories are never returned."""
    scheme, fs, path = parse_url(url)
    if fs.isfile(path):
        entries = [fs.info(path)]
    elif recursive:
        entries = list(fs.find(path, detail=True, withdirs=False).values())
    else:
        entries = [e for e in fs.ls(path, detail=True)
                   if e.get("type") != "directory"]
    sfx = tuple(s.lower() for s in suffixes)
    out = []
    for info in entries:
        name = info["name"]
        if sfx and not name.lower().endswith(sfx):
            continue
        out.append(_ref_from_info(scheme, fs, name, info))
    return sorted(out, key=lambda r: r.url)


def read_range(url: str, offset: int, length: int) -> bytes:
    """Exactly `length` bytes from `offset`, as exactly one ranged GET.

    cat_file rather than open().seek().read(): a file object is buffered,
    and its cache reads ahead. The plan and the accounting were right and
    the request was not -- a clip that planned 301,412 bytes went out as
    Range: bytes=10543362-63273573, 175 times what was counted and what a
    bucket would have billed. What a read costs is the range on the wire,
    so that is what this asks for.
    """
    _, fs, path = parse_url(url)
    if length <= 0:
        return b""
    got = fs.cat_file(path, start=int(offset), end=int(offset) + int(length))
    return bytes(got)


def put_object(url: str, data: bytes, overwrite: bool = False) -> ObjectRef:
    """Write one object, and describe it as the provider now sees it.

    The only write in the system. A pushed segment arrives as bytes over
    HTTP and has to land somewhere before it can be indexed, and every
    video byte the store later reads comes back through this same API, so
    it goes in the same way it will come out: this connector, these
    options, never a local path the store would then open.

    An existing key is refused unless asked. Raw data is immutable, and an
    accidental overwrite of an acknowledged segment would silently change
    what a query reads. The returned reference is the bucket's answer, not
    ours: its size and ETag are what a later listing will report.
    """
    _, fs, path = parse_url(url)
    if not overwrite and fs.exists(path):
        raise FileExistsError(f"{url} already exists; pass overwrite=True to replace it")
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        try:
            fs.makedirs(parent, exist_ok=True)         # object stores make this a no-op
        except (NotImplementedError, AttributeError):
            pass
    fs.pipe_file(path, bytes(data))
    fs.invalidate_cache(path)
    return stat(url)


def delete_object(url: str) -> bool:
    """Forget one object. Returns whether there was one to forget.

    The second write in the system, and the only destructive one. It exists
    because a live stream is mostly footage nobody will ever ask about: a
    robot pushes continuously, the store holds the bytes while it waits to
    be told whether they mattered, and what is never claimed has to leave.
    A store that only ever grew would make "keep the parts that mattered"
    a figure of speech.

    Raw data is still immutable -- this deletes an object, it never edits
    one. Two callers reach it: the sweep, for segments nobody claimed, and
    erasure (`erase.span`), for a stretch somebody asked to have removed.
    The second one does delete objects a manifest points at, and marks
    those rows forgotten in the same breath, because a manifest that went
    on offering seconds whose bytes had gone would be a broken answer
    rather than an honest absence. A missing object is a quiet True-less
    answer rather than an error: two sweeps racing over the same unclaimed
    segment is an ordinary thing and not a fault.
    """
    _, fs, path = parse_url(url)
    try:
        fs.rm_file(path)
    except FileNotFoundError:
        return False
    except AttributeError:                             # older fsspec filesystems
        if not fs.exists(path):
            return False
        fs.rm(path)
    finally:
        fs.invalidate_cache(path)
    return True


def fetch_scratch(url: str, scratch_dir) -> Path:
    """Copy one object to a scratch directory for an encoder that must read
    every byte (ffmpeg needs a seekable local file). The CALLER deletes the
    copy when done; nothing under scratch_dir is ever part of a store."""
    _, fs, path = parse_url(url)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dest = scratch_dir / Path(path).name
    fs.get(path, str(dest))
    return dest


def restore_request(url: str, tier: str, days: int = 2, speed: str = "Bulk"):
    """The provider call that would bring a cold object back, as data.
    None when nothing is needed. Only S3 archive classes need one today:
    GCS archive classes read directly, and Azure Archive rehydration is not
    implemented yet, so it raises rather than letting silence pass for
    success. Parsed from the URL text on purpose - no provider client is
    instantiated just to describe a request."""
    if tier != "cold":
        return None
    scheme, _, rest = url.partition("://")
    if scheme != "s3":
        raise NotImplementedError(f"restore for {scheme}:// is not implemented")
    bucket, _, key = rest.partition("/")
    return {"Bucket": bucket, "Key": key,
            "RestoreRequest": {"Days": int(days),
                               "GlacierJobParameters": {"Tier": speed}}}


def issue_restore(req: dict) -> str:
    """Send a restore request with boto3 when it is installed. Returns the
    HTTP status as text so a caller can log it; raises when boto3 is absent
    rather than pretending the restore happened."""
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 is required to issue S3 restores: "
                           "pip install boto3") from e
    r = boto3.client("s3").restore_object(**req)
    return str(r.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))
