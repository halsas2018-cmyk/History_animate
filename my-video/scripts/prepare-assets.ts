#!/usr/bin/env tsx

import { readFileSync, existsSync, statSync, createReadStream, createWriteStream } from 'fs';
import { resolve, join, basename, dirname } from 'path';
import { createHash } from 'crypto';

interface VisualTimingData {
  stage: string;
  data: {
    visuals: Array<{
      visual_index: number;
      sentence_indices: number[];
      image: string; // e.g., "scene_1.png"
      start: number;
      end: number;
      duration: number;
    }>;
  };
  meta: {
    visual_count: number;
    total_sentences: number;
  };
}

interface Args {
  [key: string]: string | boolean | number;
  'run-dir': string;
  [key: string]: unknown;
}

function parseArgs(): Args {
  const args: Args = {};
  let i = 2; // Skip node and script path

  while (i < process.argv.length) {
    const arg = process.argv[i];

    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      if (i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
        args[key] = process.argv[i + 1];
        i += 2;
      } else {
        args[key] = true;
        i++;
      }
    } else {
      // Handle positional args if needed
      i++;
    }
  }

  return args;
}

function calculateFileHash(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);

    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

async function filesAreIdentical(source: string, dest: string): Promise<boolean> {
  try {
    if (!existsSync(source) || !existsSync(dest)) {
      return false;
    }

    const sourceStat = statSync(source);
    const destStat = statSync(dest);

    // Quick check: file size
    if (sourceStat.size !== destStat.size) {
      return false;
    }

    // Check hash
    const [sourceHash, destHash] = await Promise.all([
      calculateFileHash(source),
      calculateFileHash(dest)
    ]);

    return sourceHash === destHash;
  } catch (error) {
    return false;
  }
}

async function copyFileIfDifferent(source: string, dest: string): Promise<void> {
  try {
    // Ensure destination directory exists
    const destDir = dirname(dest);
    if (!existsSync(destDir)) {
      // Simple approach - we'll let the copy operation handle directory creation
      // or fail with a clear error
    }

    // For simplicity in this implementation, we'll just copy
    const sourceStream = createReadStream(source);
    const destStream = createWriteStream(dest);
    sourceStream.pipe(destStream);

    return new Promise((resolve, reject) => {
      sourceStream.on('error', reject);
      destStream.on('error', reject);
      destStream.on('finish', resolve);
    });
  } catch (error) {
    throw new Error(`Failed to copy ${source} to ${dest}: ${error.message}`);
  }
}

async function main() {
  try {
    const args = parseArgs();
    const runDirArg = args['run-dir'];

    if (!runDirArg || typeof runDirArg !== 'string') {
      console.error('Error: --run-dir argument is required');
      process.exit(1);
    }

    const runDir = resolve(runDirArg);
    console.log(`Using run directory: ${runDir}`);

    // Validate run directory exists
    if (!existsSync(runDir)) {
      console.error(`Error: Run directory does not exist: ${runDir}`);
      process.exit(1);
    }

    // Validate visual_timing.json exists
    const visualTimingPath = join(runDir, 'visual_timing.json');
    if (!existsSync(visualTimingPath)) {
      console.error(`Error: visual_timing.json not found in run directory: ${visualTimingPath}`);
      process.exit(1);
    }

    // Validate narration.mp3 exists
    const narrationPath = join(runDir, 'narration.mp3');
    if (!existsSync(narrationPath)) {
      console.error(`Error: narration.mp3 not found in run directory: ${narrationPath}`);
      process.exit(1);
    }

    // Validate whiteboard/ directory exists
    const whiteboardDir = join(runDir, 'whiteboard');
    if (!existsSync(whiteboardDir)) {
      console.error(`Error: whiteboard directory not found in run directory: ${whiteboardDir}`);
      process.exit(1);
    }

    // Parse visual_timing.json
    let visualTimingData: VisualTimingData;
    try {
      const fileContent = readFileSync(visualTimingPath, 'utf8');
      visualTimingData = JSON.parse(fileContent);
    } catch (error) {
      console.error(`Error: Failed to parse visual_timing.json: ${error.message}`);
      process.exit(1);
    }

    // Validate structure
    if (!visualTimingData.data || !Array.isArray(visualTimingData.data.visuals)) {
      console.error('Error: Invalid visual_timing.json structure - missing or invalid data.visuals array');
      process.exit(1);
    }

    const visuals = visualTimingData.data.visuals;
    console.log(`Found ${visuals.length} visual entries in timing data`);

    // Extract required scene filenames
    const requiredScenes: string[] = [];
    for (const visual of visuals) {
      if (!visual.image || typeof visual.image !== 'string') {
        console.error('Error: Invalid visual entry - missing or invalid image field');
        process.exit(1);
      }

      // Convert image filename to MP4 filename by replacing extension
      // and converting underscores to spaces to match actual filenames (scene_1.png -> scene 1.mp4)
      const mp4Filename = visual.image
        .replace(/\.[^.]+$/, '')  // Remove extension
        .replace(/_/g, ' ') + '.mp4'; // Replace underscores with spaces and add .mp4
      requiredScenes.push(mp4Filename);
    }

    // Verify each required scene MP4 exists in whiteboard/
    const missingScenes: string[] = [];
    for (const scene of requiredScenes) {
      const scenePath = join(whiteboardDir, scene);
      if (!existsSync(scenePath)) {
        missingScenes.push(scene);
      }
    }

    if (missingScenes.length > 0) {
      console.error(`Error: Missing scene MP4 files in whiteboard/: ${missingScenes.join(', ')}`);
      process.exit(1);
    }

    // Prepare public directory
    const publicDir = join(process.cwd(), 'public');
    if (!existsSync(publicDir)) {
      console.error(`Error: public directory does not exist: ${publicDir}`);
      process.exit(1);
    }

    // Copy narration.mp3
    const destNarrationPath = join(publicDir, 'narration.mp3');
    console.log(`Copying narration.mp3...`);

    if (await filesAreIdentical(narrationPath, destNarrationPath)) {
      console.log(`  Skipped identical narration.mp3`);
    } else {
      await copyFileIfDifferent(narrationPath, destNarrationPath);
      console.log(`  Copied narration.mp3`);
    }

    // Copy scene MP4s
    let copiedCount = 0;
    let skippedCount = 0;

    for (const scene of requiredScenes) {
      const sourceScenePath = join(whiteboardDir, scene);
      const destScenePath = join(publicDir, scene);

      if (await filesAreIdentical(sourceScenePath, destScenePath)) {
        console.log(`  Skipped identical ${scene}`);
        skippedCount++;
      } else {
        await copyFileIfDifferent(sourceScenePath, destScenePath);
        console.log(`  Copied ${scene}`);
        copiedCount++;
      }
    }

    // Print summary
    console.log(`\nPrepared ${requiredScenes.length} scene MP4s and narration.mp3 for Remotion.`);
    console.log(`  Copied: ${copiedCount} scenes, 1 narration`);
    console.log(`  Skipped (identical): ${skippedCount} scenes`);

  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main().catch((error) => {
  console.error(`Unexpected error: ${error.message}`);
  process.exit(1);
});