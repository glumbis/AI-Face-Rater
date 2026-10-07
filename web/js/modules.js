// The one place where the app picks its feature modules. Each line imports either the real module or its stub from
// ./stubs/. When a real module lands, change that one path (for example './stubs/scoring.js' -> './scoring.js').
export * as scoring from './stubs/scoring.js';
export * as headpose from './stubs/headpose.js';
export * as facedata from './stubs/facedata.js';
export * as tips from './stubs/tips.js';
export * as history from './stubs/history.js';
export * as share from './stubs/share.js';
