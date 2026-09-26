/** The one eye: open and red while a camera is live. */
export function Eye({ live = false }: { live?: boolean }) {
  return (
    <svg className={live ? "eye live" : "eye"} viewBox="0 0 32 32" aria-hidden="true">
      <path d="M2 16C7 8 11 6 16 6s9 2 14 10c-5 8-9 10-14 10S7 24 2 16z" />
      <circle className="iris" cx="16" cy="16" r="5" />
    </svg>
  );
}
