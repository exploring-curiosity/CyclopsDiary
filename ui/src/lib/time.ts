// World time as the diary's answers word it (router._t): this machine's own clock, to the second.

export const secs = (iso: string) => Date.parse(iso) / 1000;

const two = (n: number) => String(n).padStart(2, "0");

/** "11:10:20 am" */
export function clock(at: string | number) {
  const d = new Date(typeof at === "number" ? at * 1000 : at);
  return `${d.getHours() % 12 || 12}:${two(d.getMinutes())}:${two(d.getSeconds())} ${d.getHours() < 12 ? "am" : "pm"}`;
}

/** A stretch of time between recordings, short: "+40 s", "+12 min", "+3 h 5 min", "+2 days". */
export function gap(s: number) {
  if (s < 90) return `+${Math.round(s)} s`;
  if (s < 5400) return `+${Math.round(s / 60)} min`;
  if (s < 172800) {
    const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
    return m ? `+${h} h ${m} min` : `+${h} h`;
  }
  return `+${Math.round(s / 86400)} days`;
}
