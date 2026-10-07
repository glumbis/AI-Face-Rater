// The one place where the app picks its feature modules. Each line imports either the real module or its stub from
// ./stubs/. When a real module lands, change that one path (for example './stubs/scoring.js' -> './scoring.js').
export * as scoring from './scoring.js';
export * as headpose from './headpose.js';
export * as facedata from './facedata.js';
// The real tips.js imports ./facedata.js, so it can only replace its stub once the generated facedata.js exists
export * as tips from './tips.js';
export * as history from './history.js';
export * as share from './share.js';
