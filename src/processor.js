export class WhitespaceTokenSplitter {
    constructor() {
        this.whitespacePattern = /\w+(?:[-_]\w+)*|\S/g;
    }

    *call(text) {
        let match;
        while ((match = this.whitespacePattern.exec(text)) !== null) {
            yield [match[0], match.index, this.whitespacePattern.lastIndex];
        }
    }
}

export class Processor { 
    constructor(config, tokenizer, wordsSplitter) {
        this.config = config;
        this.tokenizer = tokenizer;
        this.wordsSplitter = wordsSplitter;
    }

    tokenizeText(text) {
        let tokens = [];
        let wordsStartIdx = [];
        let wordsEndIdx = [];

        for (const [token, start, end] of this.wordsSplitter.call(text)) {
            tokens.push(token);
            wordsStartIdx.push(start);
            wordsEndIdx.push(end);
        }
        return [tokens, wordsStartIdx, wordsEndIdx];
    }

    batchTokenizeText(texts) {
        let batchTokens = [];
        let batchWordsStartIdx = [];
        let batchWordsEndIdx = [];

        for (const text of texts) {  // Use 'for...of' instead of 'for...in'
            const [tokens, wordsStartIdx, wordsEndIdx] = this.tokenizeText(text);

            batchTokens.push(tokens);
            batchWordsStartIdx.push(wordsStartIdx);
            batchWordsEndIdx.push(wordsEndIdx);
        }

        return [batchTokens, batchWordsStartIdx, batchWordsEndIdx];
    }

    createMappings(classes) {
        const classToId = {};
        const idToClass = {};
    
        classes.forEach((className, index) => {
            const id = index + 1;  // Start numbering from 1
            classToId[className] = id;
            idToClass[id] = className;
        });
    
        return  { classToId, idToClass };
    }

    prepareTextInputs(tokens, entities) {
        const inputTexts = [];
        const promptLengths = [];
        const textLengths = [];
        tokens.forEach((text, id) => {
            textLengths.push(text.length);  // Corrected typo 'textLengths'

            let inputText = [];
            for (let ent of entities) {
                inputText.push("<<ENT>>");
                inputText.push(ent);
            }
            inputText.push('<<SEP>>');
            const promptLength = inputText.length;
            promptLengths.push(promptLength);
            inputText = inputText.concat(text);
            inputTexts.push(inputText);
        });
        return [inputTexts, textLengths, promptLengths];
    }

    encodeInputs(texts, promptLengths = null) {
        let wordsMasks = [];
        let inputsIds = [];
        let attentionMasks = [];
        for (let id = 0; id < texts.length; id++) {
            let promptLength = promptLengths ? promptLengths[id] : 0;
            let tokenizedInputs = texts[id];
            let wordsMask = [0];        
            let inputIds = [1];
            let attentionMask = [1];

            tokenizedInputs.forEach((word, wordId) => {
                let wordTokens = this.tokenizer.encode(word).slice(1, -1);  // Use this.tokenizer
                wordTokens.forEach((token, tokenId) => {
                    attentionMask.push(1);
                    if (wordId < promptLength) {
                        wordsMask.push(0);
                    }
                    else if (tokenId === 0) {
                        wordsMask.push(1);
                    }
                    else {
                        wordsMask.push(0);
                    }
                    inputIds.push(token);
                });
            });
            wordsMask.push(0);
            inputIds.push(this.tokenizer.sep_token_id);  // Use this.tokenizer
            attentionMask.push(1);
    
            wordsMasks.push(wordsMask);
            inputsIds.push(inputIds);
            attentionMasks.push(attentionMask);
        }
        return [inputsIds, attentionMasks, wordsMasks];
    }

    padArray(arr, dimensions = 2) {
        if (dimensions < 2 || dimensions > 3) {
            throw new Error('Only 2D and 3D arrays are supported');
        }

        const maxLength = Math.max(...arr.map(subArr => subArr.length));
        const finalDim = dimensions === 3 ? arr[0][0].length : 0;

        return arr.map(subArr => {
            const padCount = maxLength - subArr.length;
            const padding = Array(padCount).fill(
            dimensions === 3 ? Array(finalDim).fill(0) : 0
            );
            return [...subArr, ...padding];
        });
    }

}

export class SpanProcessor extends Processor {
    constructor(config, tokenizer, wordsSplitter) {
        super(config, tokenizer, wordsSplitter);  // Call super
    }

    prepareSpans(batchTokens, maxWidth = 12) {
        let spanIdxs = [];
        let spanMasks = [];
      
        batchTokens.forEach((tokens) => {
          let textLength = tokens.length;
          let spanIdx = [];
          let spanMask = [];
      
          for (let i = 0; i < textLength; i++) {
            for (let j = 1; j <= maxWidth; j++) {
              let endIdx = Math.min(i + j, textLength-1);
              spanIdx.push([i, endIdx]);
              spanMask.push(endIdx <= textLength ? 1 : 0);
            }
          }
          // Pad spans if necessary
          const requiredSpans = textLength * maxWidth;
          while (spanIdx.length < requiredSpans) {
            spanIdx.push([textLength - 1, textLength]);
            spanMask.push(0);
          }
      
          spanIdxs.push(spanIdx);
          spanMasks.push(spanMask);
        });
      
        return { spanIdxs, spanMasks };
      }      

    prepareBatch(texts, entities) {
        const [batchTokens, batchWordsStartIdx, batchWordsEndIdx] = this.batchTokenizeText(texts);

        const { classToId, idToClass } = this.createMappings(entities);

        const [inputTokens, textLengths, promptLengths] = this.prepareTextInputs(batchTokens, entities);

        let [inputsIds, attentionMasks, wordsMasks] = this.encodeInputs(inputTokens, promptLengths);
        
        inputsIds = this.padArray(inputsIds);
        attentionMasks = this.padArray(attentionMasks);
        wordsMasks = this.padArray(wordsMasks);

        let { spanIdxs, spanMasks } = this.prepareSpans(batchTokens, this.config["max_width"]);

        spanIdxs = this.padArray(spanIdxs, 3);
        spanMasks = this.padArray(spanMasks);

        return {
            inputsIds: inputsIds,
            attentionMasks: attentionMasks,
            wordsMasks: wordsMasks,
            textLengths: textLengths,
            spanIdxs: spanIdxs,
            spanMasks: spanMasks,
            idToClass: idToClass,
            batchTokens: batchTokens,
            batchWordsStartIdx: batchWordsStartIdx,
            batchWordsEndIdx: batchWordsEndIdx, 
        };
    }
}