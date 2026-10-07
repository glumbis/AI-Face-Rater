// STUB. Replaced by the real js/share.js.
export async function makeShareCard({ score, reference, theme }) {
  const c = document.createElement('canvas'); c.width = 1080; c.height = 1080;
  const g = c.getContext('2d');
  g.fillStyle = theme === 'dark' ? '#202020' : '#f3f3f3'; g.fillRect(0, 0, 1080, 1080);
  g.fillStyle = theme === 'dark' ? '#f5f5f5' : '#1a1a1a'; g.font = '600 280px system-ui'; g.textAlign = 'center';
  g.fillText(score.toFixed(1), 540, 560);
  g.font = '40px system-ui'; g.fillText(`${reference} reference`, 540, 660);
  return new Promise((r) => c.toBlob(r, 'image/png'));
}
export async function shareOrDownload(blob) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'face-rater.png'; a.click();
}
