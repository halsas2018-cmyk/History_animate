/**
 * Note: When using the Node.JS APIs, the config file
 * doesn't apply. Instead, pass options directly to the APIs.
 *
 * All configuration options: https://remotion.dev/docs/config
 */

import { Config } from "@remotion/cli/config";

Config.setRspack(true);
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);

// Use ANGLE software WebGL for headless rendering.
// Without a GPU on this machine, Chromium's default headless WebGL2
// backend can't create a context within Remotion's 28s `delayRender`
// window, causing <ThreeCanvas> to time out during full video renders.
//
// Setting `ChromiumOpenGlRenderer = "angle"` enables the SwiftShader
// software WebGL implementation, which is slow but works on any CPU.
//
// If you ever run this on a machine with a real GPU, change this to
// `"desktop"` (or remove the call entirely) for much faster renders.
Config.setChromiumOpenGlRenderer("angle");

// Increase the per-frame delayRender timeout from the default 28s to 2 minutes.
// Software WebGL (SwiftShader) on CPU is significantly slower than hardware
// WebGL. The first frame in particular can take a while to compile shaders
// and upload geometry. 28s is not enough margin for a 203-cube scene on CPU.
//
// 2 minutes (120_000 ms) is plenty for any frame in this project.
//
// If you ever switch to hardware WebGL (Config.setChromiumOpenGlRenderer("desktop")
// on a machine with a GPU), you can lower this back to 28_000.
Config.setDelayRenderTimeoutInMilliseconds(120_000);
