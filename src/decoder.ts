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
const hasOverlappingNested = (idx1: number[], idx2: number[], multiLabel: boolean = false): boolean => {
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

    greedySearch(spans: number[][], flatNer: boolean = true, multiLabel: boolean = false): number[][] {
        const hasOv = flatNer
            ? (idx1: number[], idx2: number[]) => hasOverlapping(idx1, idx2, multiLabel)
            : (idx1: number[], idx2: number[]) => hasOverlappingNested(idx1, idx2, multiLabel);

        const newList: number[][] = [];
        const spanProb = spans.slice().sort((a, b) => b[b.length - 1] - a[a.length - 1]); // Sort by score

        for (let i = 0; i < spans.length; i++) {
            const b = spanProb[i];
            let flag = false;
            for (const newSpan of newList) {
                if (hasOv(b.slice(0, -1), newSpan)) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                newList.push(b);
            }
        }

        return newList.sort((a, b) => a[0] - b[0]); // Sort by start position
    }
}

// SpanDecoder subclass
export class SpanDecoder extends BaseDecoder {
    decode(
        batchSize: number,
        inputLength: number,
        maxWidth: number,
        numEntities: number,
        texts: string[][],
        batchIds: number[],
        batchWordsStartIdx: number[][],
        batchWordsEndIdx: number[][],
        idToClass: Record<number, string>,
        modelOutput: number[],
        flatNer: boolean = false,
        threshold: number = 0.5,
        multiLabel: boolean = false
    ): number[][][] {
        const spans: any[][][] = [];

        for (let batch = 0; batch < batchSize; batch++) {
            spans.push([]);
        }

        const batchPadding = inputLength * maxWidth * numEntities;
        const startTokenPadding = maxWidth * numEntities;
        const endTokenPadding = numEntities * 1;

        modelOutput.forEach((value, id) => {
            let batch = Math.floor(id / batchPadding);
            let startToken = Math.floor(id / startTokenPadding) % inputLength;
            let endToken = startToken + Math.floor(id / endTokenPadding) % maxWidth;
            let entity = id % numEntities;

            let prob = sigmoid(value);

            if (
                prob >= threshold &&
                startToken < batchWordsStartIdx[batch].length &&
                endToken < batchWordsEndIdx[batch].length
            ) {
                let globalBatch = batchIds[batch];
                let startIdx = batchWordsStartIdx[batch][startToken];
                let endIdx = batchWordsEndIdx[batch][endToken];
                let spanText = texts[globalBatch].slice(startIdx, endIdx);
                spans[batch].push([
                    spanText,
                    startIdx,
                    endIdx,
                    idToClass[entity + 1],
                    prob
                ]);
            }
        });
        const allSelectedSpans: number[][][] = [];

        spans.forEach((resI, id) => {
            const selectedSpans = this.greedySearch(resI, flatNer, multiLabel);
            allSelectedSpans.push(selectedSpans);
        });

        return spans;
    }
}
