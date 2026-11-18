/** Copy text; works on plain-http pages where navigator.clipboard is missing. Falls back to a prompt the user can copy from. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    if (ok) return true;
  } catch { /* fall through */ }
  window.prompt("Copy this link (Ctrl+C), then press Enter:", text);
  return false;
}
