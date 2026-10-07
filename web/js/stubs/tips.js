// STUB. Replaced by the real js/tips.js.
export function photoTips(imageData, points, angles) {
  const tips = [];
  if (angles && Math.abs(angles.turn) > 12) tips.push('Your head is turned a little. Look straight at the lens.');
  if (angles && Math.abs(angles.tilt) > 15) tips.push('Your head is tilted a little. Look straight at the lens.');
  return tips.length ? tips.slice(0, 2) : ['Great setup: straight, sharp and evenly lit.'];
}
