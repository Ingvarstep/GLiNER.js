
// Check if one span is nested inside the other
const isNested = (idx1, idx2) => {
    return (idx1[0] <= idx2[0] && idx1[1] >= idx2[1]) || (idx2[0] <= idx1[0] && idx2[1] >= idx1[1]);
};

// Check for any overlap between two spans
const hasOverlapping = (idx1, idx2, multiLabel = false) => {
    // If the exact same boundaries exist, consider as overlapping unless multiLabel is true
    if (idx1.slice(0, 2).toString() === idx2.slice(0, 2).toString()) {
        return !multiLabel;
    }
    // No overlap if one span starts after the other ends
    if (idx1[0] > idx2[1] || idx2[0] > idx1[1]) {
        return false;
    }
    return true;
};

// Check if spans overlap but are not nested inside each other
const hasOverlappingNested = (idx1, idx2, multiLabel = false) => {
    // If the exact same boundaries exist, consider as overlapping unless multiLabel is true
    if (idx1.slice(0, 2).toString() === idx2.slice(0, 2).toString()) {
        return !multiLabel;
    }
    // No overlap if spans are either non-overlapping or nested
    if (idx1[0] > idx2[1] || idx2[0] > idx1[1] || isNested(idx1, idx2)) {
        return false;
    }
    return true;
};

const sigmoid = (x) => {
    return 1 / (1 + Math.exp(-x));
};

// BaseDecoder class using abstract methods
class BaseDecoder {
    constructor(config) {
        if (new.target === BaseDecoder) {
            throw new TypeError("Cannot instantiate an abstract class.");
        }
        this.config = config;
    }

    // Abstract method
    decode(...args) {
        throw new Error("Abstract method 'decode' must be implemented by subclasses.");
    }

    // Greedy search method
    greedySearch(spans, flatNer = true, multiLabel = false) {
        const hasOv = flatNer
            ? (idx1, idx2) => hasOverlapping(idx1, idx2, multiLabel)
            : (idx1, idx2) => hasOverlappingNested(idx1, idx2, multiLabel);

        const newList = [];
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
    decode(tokens, batchWordsStartIdx, batchWordsEndIdx, idToClasses, 
                modelOutput, flatNer = false, threshold = 0.5, multiLabel = false) {
        const spans = [];

        tokens.forEach((currTokens, i) => {
            const outputI = modelOutput[i];
            const resI = [];
            
            let WordsStartIdx = batchWordsStartIdx[i];
            let WordsEndIdx = batchWordsEndIdx[i];

            outputI.forEach((tokensI, s)=> {
                tokensI.forEach((spanI, k) => {
                    spanI.forEach((classI, c)=> {
                        let prob = sigmoid(classI);
                        if (prob>=threshold) {
                            resI.push([WordsStartIdx[s], WordsEndIdx[s + k], 
                                                    idToClasses[c + 1], prob]);
                        }
                    });
                });
            });

            const selectedSpans = this.greedySearch(resI, flatNer, multiLabel);
            spans.push(selectedSpans);
        });

        return spans;
    }
}