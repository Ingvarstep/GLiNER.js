import { RawInferenceResult } from "./Gliner";

type Spans = [string, number, number, string, number][];

// Check if one span is nested inside the other
const isNested = (idx1: number[], idx2: number[]): boolean => {
  return (idx1[0] <= idx2[0] && idx1[1] >= idx2[1]) || (idx2[0] <= idx1[0] && idx2[1] >= idx1[1]);
};

// Check for any overlap between two spans
const hasOverlapping = (idx1: number[], idx2: number[], multiLabel: boolean = false): boolean => {
  if (idx1.slice(0, 2).toString() === idx2.slice(0, 2).toString()) {
    return !multiLabel;
  }
  if (idx1[0] > idx2[1] || idx2[0] > idx1[1]) {
    return false;
  }
  return true;
};

// Check if spans overlap but are not nested inside each other
const hasOverlappingNested = (
  idx1: number[],
  idx2: number[],
  multiLabel: boolean = false,
): boolean => {
  if (idx1.slice(0, 2).toString() === idx2.slice(0, 2).toString()) {
    return !multiLabel;
  }
  if (idx1[0] > idx2[1] || idx2[0] > idx1[1] || isNested(idx1, idx2)) {
    return false;
  }
  return true;
};

const sigmoid = (x: number): number => {
  return 1 / (1 + Math.exp(-x));
};

// BaseDecoder class using abstract methods
abstract class BaseDecoder {
  config: any;

  constructor(config: any) {
    if (new.target === BaseDecoder) {
      throw new TypeError("Cannot instantiate an abstract class.");
    }
    this.config = config;
  }

  abstract decode(...args: any[]): any;

  greedySearch(spans: Spans, flatNer: boolean = true, multiLabel: boolean = false): Spans {
    const hasOv = flatNer
      ? (idx1: number[], idx2: number[]) => hasOverlapping(idx1, idx2, multiLabel)
      : (idx1: number[], idx2: number[]) => hasOverlappingNested(idx1, idx2, multiLabel);

    const newList: Spans = [];
    // Sort spans by their score (last element) in descending order
    const spanProb: Spans = spans.slice().sort((a, b) => b[b.length - 1] - a[a.length - 1]);

    for (let i = 0; i < spanProb.length; i++) {
      const b = spanProb[i];
      let flag = false;
      for (const newSpan of newList) {
        // Compare only start and end indices
        if (hasOv(b.slice(1, 3), newSpan.slice(1, 3))) {
          flag = true;
          break;
        }
      }
      if (!flag) {
        newList.push(b);
      }
    }

    // Sort newList by start position (second element) for correct ordering
    return newList.sort((a, b) => a[1] - b[1]);
  }
}
// SpanDecoder subclass
export class SpanDecoder extends BaseDecoder {
  decode(
    batchSize: number,
    inputLength: number,
    maxWidth: number,
    numEntities: number,
    texts: string[],
    batchIds: number[],
    batchWordsStartIdx: number[][],
    batchWordsEndIdx: number[][],
    idToClass: Record<number, string>,
    modelOutput: number[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
  ): RawInferenceResult {
    const spans: RawInferenceResult = [];

    for (let batch = 0; batch < batchSize; batch++) {
      spans.push([]);
    }

    const batchPadding = inputLength * maxWidth * numEntities;
    const startTokenPadding = maxWidth * numEntities;
    const endTokenPadding = numEntities * 1;

    modelOutput.forEach((value, id) => {
      let batch = Math.floor(id / batchPadding);
      let startToken = Math.floor(id / startTokenPadding) % inputLength;
      let endToken = startToken + (Math.floor(id / endTokenPadding) % maxWidth);
      let entity = id % numEntities;

      let prob = sigmoid(value);

      if (
        prob >= threshold &&
        startToken < batchWordsStartIdx[batch].length &&
        endToken < batchWordsEndIdx[batch].length
      ) {
        let globalBatch: number = batchIds[batch];
        let startIdx: number = batchWordsStartIdx[batch][startToken];
        let endIdx: number = batchWordsEndIdx[batch][endToken];
        let spanText: string = texts[globalBatch].slice(startIdx, endIdx);
        spans[batch].push([spanText, startIdx, endIdx, idToClass[entity + 1], prob]);
      }
    });
    const allSelectedSpans: RawInferenceResult = [];

    spans.forEach((resI, id) => {
      const selectedSpans = this.greedySearch(resI, flatNer, multiLabel);
      allSelectedSpans.push(selectedSpans);
    });

    return allSelectedSpans;
  }
}

// TokenDecoder subclass
export class TokenDecoder extends BaseDecoder {
  decode(
    batchSize: number,
    inputLength: number,
    numEntities: number,
    texts: string[],
    batchIds: number[],
    batchWordsStartIdx: number[][],
    batchWordsEndIdx: number[][],
    idToClass: Record<number, string>,
    modelOutput: number[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
  ): RawInferenceResult {
    const positionPadding = batchSize * inputLength * numEntities;
    const batchPadding = inputLength * numEntities;
    const tokenPadding = numEntities;

    let selectedStarts: any = [];
    let selectedEnds: any = [];
    let insideScore: number[][][] = [];

    for (let id = 0; id < batchSize; id++) {
      selectedStarts.push([]);
      selectedEnds.push([]);
      let batches: number[][] = [];
      for (let j = 0; j < inputLength; j++) {
        let sequence: number[] = Array(numEntities).fill(0);
        batches.push(sequence);
      }
      insideScore.push(batches);
    }

    modelOutput.forEach((value, id) => {
      let position = Math.floor(id / positionPadding);
      let batch = Math.floor(id / batchPadding) % batchSize;
      let token = Math.floor(id / tokenPadding) % inputLength;
      let entity = id % numEntities;

      let prob = sigmoid(value);

      if (prob >= threshold && token < batchWordsEndIdx[batch].length) {
        if (position == 0) {
          selectedStarts[batch].push([token, entity]);
        } else if (position == 1) {
          selectedEnds[batch].push([token, entity]);
        }
      }
      if (position == 2) {
        insideScore[batch][token][entity] = prob;
      }
    });

    const spans: RawInferenceResult = [];

    for (let batch = 0; batch < batchSize; batch++) {
      let batchSpans: Spans = [];

      for (let [start, clsSt] of selectedStarts[batch]) {
        for (let [end, clsEd] of selectedEnds[batch]) {
          if (end >= start && clsSt === clsEd) {
            // Calculate the inside span scores
            const insideSpanScores = insideScore[batch]
              .slice(start, end + 1)
              .map((tokenScores) => tokenScores[clsSt]);

            // Check if all scores within the span are above the threshold
            if (insideSpanScores.some((score) => score < threshold)) continue;

            // Calculate mean span score
            const spanScore = insideSpanScores.reduce((a, b) => a + b, 0) / insideSpanScores.length;

            // Extract the start and end indices and the text for the span
            let startIdx = batchWordsStartIdx[batch][start];
            let endIdx = batchWordsEndIdx[batch][end];
            let spanText = texts[batchIds[batch]].slice(startIdx, endIdx);

            // Push the span with its score and class
            batchSpans.push([spanText, startIdx, endIdx, idToClass[clsSt + 1], spanScore]);
          }
        }
      }
      spans.push(batchSpans);
    }

    const allSelectedSpans: RawInferenceResult = [];

    spans.forEach((resI, id) => {
      const selectedSpans = this.greedySearch(resI, flatNer, multiLabel);
      allSelectedSpans.push(selectedSpans);
    });

    return allSelectedSpans;
  }
}
