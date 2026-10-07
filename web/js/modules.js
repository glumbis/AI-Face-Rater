// The one place where the app picks its feature modules. Each line imports either the real module or its stub from
// ./stubs/. When a real module lands, change that one path (for example './stubs/scoring.js' -> './scoring.js').
export * as scoring from './stubs/scoring.js';   // TODO real ./scoring.js (scoring agent)
export * as headpose from './stubs/headpose.js'; // TODO real ./headpose.js (scoring agent)
export * as facedata from './stubs/facedata.js'; // TODO real ./facedata.js (generated)
// The real tips.js imports ./facedata.js, so it can only replace its stub once the generated facedata.js exists
export * as tips from './stubs/tips.js';         // TODO real ./tips.js together with facedata.js
export * as history from './history.js';
export * as share from './share.js';
