"""Settings every test in this directory needs before anything imports.

Set here rather than in pytest.ini because the `env` ini option belongs
to pytest-env, which is not a dependency of this project: a setting that
is silently ignored when a plugin is missing is worse than no setting.
"""
import os

# The encoder's URL conf warms the tower a store's trace is pinned to
# when the process starts, so a cold container does not pay a
# multi-gigabyte download on its first question. Tests run against a stub
# tower and must not pull a real one into the background while they
# assert on what is resident.
os.environ.setdefault("ELIDEDB_WARM_ON_START", "0")

# A test must never write into the deployment's real lake. `lake_root()`
# falls back to <repo>/data/relmo/lake when ELIDEDB_LAKE is unset, and a
# django test that claimed a store through it left an empty store behind
# in the operator's own data -- which then appeared in the console as a
# real store with nothing in it. Tests that want their own lake still set
# the variable themselves; this only replaces the fallback.
import tempfile

os.environ.setdefault("ELIDEDB_LAKE", tempfile.mkdtemp(prefix="elidedb-test-lake-"))
