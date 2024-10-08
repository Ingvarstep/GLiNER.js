export class WhitespaceTokenSplitter {
  private whitespacePattern: RegExp;

  constructor() {
    this.whitespacePattern = /\w+(?:[-_]\w+)*|\S/g;
  }

  *call(text: string): Generator<[string, number, number]> {
    let match: RegExpExecArray | null;
    while ((match = this.whitespacePattern.exec(text)) !== null) {
      yield [match[0], match.index, this.whitespacePattern.lastIndex];
    }
  }
}

export class Processor {
  config: any;
  tokenizer: any;
  wordsSplitter: WhitespaceTokenSplitter;

  constructor(config: any, tokenizer: any, wordsSplitter: WhitespaceTokenSplitter) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.wordsSplitter = wordsSplitter;
  }

  tokenizeText(text: string): [string[], number[], number[]] {
    let tokens: string[] = [];
    let wordsStartIdx: number[] = [];
    let wordsEndIdx: number[] = [];

    for (const [token, start, end] of this.wordsSplitter.call(text)) {
      tokens.push(token);
      wordsStartIdx.push(start);
      wordsEndIdx.push(end);
    }
    return [tokens, wordsStartIdx, wordsEndIdx];
  }

  batchTokenizeText(texts: string[]): [string[][], number[][], number[][]] {
    let batchTokens: string[][] = [];
    let batchWordsStartIdx: number[][] = [];
    let batchWordsEndIdx: number[][] = [];

    for (const text of texts) {
      const [tokens, wordsStartIdx, wordsEndIdx] = this.tokenizeText(text);
      batchTokens.push(tokens);
      batchWordsStartIdx.push(wordsStartIdx);
      batchWordsEndIdx.push(wordsEndIdx);
    }

    return [batchTokens, batchWordsStartIdx, batchWordsEndIdx];
  }

  createMappings(classes: string[]): {
    classToId: Record<string, number>;
    idToClass: Record<number, string>;
  } {
    const classToId: Record<string, number> = {};
    const idToClass: Record<number, string> = {};

    classes.forEach((className, index) => {
      const id = index + 1; // Start numbering from 1
      classToId[className] = id;
      idToClass[id] = className;
    });

    return { classToId, idToClass };
  }

  prepareTextInputs(tokens: string[][], entities: string[]): [string[][], number[], number[]] {
    const inputTexts: string[][] = [];
    const promptLengths: number[] = [];
    const textLengths: number[] = [];

    tokens.forEach((text) => {
      textLengths.push(text.length);

      let inputText: string[] = [];
      for (let ent of entities) {
        inputText.push("<<ENT>>");
        inputText.push(ent);
      }
      inputText.push("<<SEP>>");
      const promptLength = inputText.length;
      promptLengths.push(promptLength);
      inputText = inputText.concat(text);
      inputTexts.push(inputText);
    });
    return [inputTexts, textLengths, promptLengths];
  }

  encodeInputs(
    texts: string[][],
    promptLengths: number[] | null = null,
  ): [number[][], number[][], number[][]] {
    let wordsMasks: number[][] = [];
    let inputsIds: number[][] = [];
    let attentionMasks: number[][] = [];

    for (let id = 0; id < texts.length; id++) {
      let promptLength = promptLengths ? promptLengths[id] : 0;
      let tokenizedInputs = texts[id];
      let wordsMask: number[] = [0];
      let inputIds: number[] = [1];
      let attentionMask: number[] = [1];

      let c = 1;
      tokenizedInputs.forEach((word, wordId) => {
        let wordTokens = this.tokenizer.encode(word).slice(1, -1);
        wordTokens.forEach((token: number, tokenId: number) => {
          attentionMask.push(1);
          if (wordId < promptLength) {
            wordsMask.push(0);
          } else if (tokenId === 0) {
            wordsMask.push(c);
            c++;
          } else {
            wordsMask.push(0);
          }
          inputIds.push(token);
        });
      });
      wordsMask.push(0);
      inputIds.push(this.tokenizer.sep_token_id);
      attentionMask.push(1);

      wordsMasks.push(wordsMask);
      inputsIds.push(inputIds);
      attentionMasks.push(attentionMask);
    }
    return [inputsIds, attentionMasks, wordsMasks];
  }

  padArray(arr: any[], dimensions: number = 2): any[][] {
    if (dimensions < 2 || dimensions > 3) {
      throw new Error("Only 2D and 3D arrays are supported");
    }

    const maxLength: number = Math.max(...arr.map((subArr: any[]) => subArr.length));
    const finalDim: number = dimensions === 3 ? arr[0][0].length : 0;

    return arr.map((subArr: any[]) => {
      const padCount = maxLength - subArr.length;
      const padding = Array(padCount).fill(dimensions === 3 ? Array(finalDim).fill(0) : 0);
      return [...subArr, ...padding];
    });
  }
}

export class SpanProcessor extends Processor {
  constructor(config: any, tokenizer: any, wordsSplitter: WhitespaceTokenSplitter) {
    super(config, tokenizer, wordsSplitter);
  }

  prepareSpans(
    batchTokens: string[][],
    maxWidth: number = 12,
  ): { spanIdxs: number[][][]; spanMasks: boolean[][] } {
    let spanIdxs: number[][][] = [];
    let spanMasks: boolean[][] = [];

    batchTokens.forEach((tokens) => {
      let textLength: number = tokens.length;
      let spanIdx: number[][] = [];
      let spanMask: boolean[] = [];

      for (let i = 0; i < textLength; i++) {
        for (let j = 0; j < maxWidth; j++) {
          let endIdx = Math.min(i + j, textLength - 1);
          spanIdx.push([i, endIdx]);
          spanMask.push(endIdx < textLength ? true : false);
        }
      }

      spanIdxs.push(spanIdx);
      spanMasks.push(spanMask);
    });

    return { spanIdxs, spanMasks };
  }

  prepareBatch(texts: string[], entities: string[]): Record<string, any> {
    const [batchTokens, batchWordsStartIdx, batchWordsEndIdx]: [
      string[][],
      number[][],
      number[][],
    ] = this.batchTokenizeText(texts);
    const { idToClass }: { idToClass: Record<number, string> } = this.createMappings(entities);
    const [inputTokens, textLengths, promptLengths]: [string[][], number[], number[]] =
      this.prepareTextInputs(batchTokens, entities);

    let [inputsIds, attentionMasks, wordsMasks]: [number[][], number[][], number[][]] =
      this.encodeInputs(inputTokens, promptLengths);

    inputsIds = this.padArray(inputsIds);
    attentionMasks = this.padArray(attentionMasks);
    wordsMasks = this.padArray(wordsMasks);

    let { spanIdxs, spanMasks }: { spanIdxs: number[][][]; spanMasks: boolean[][] } =
      this.prepareSpans(batchTokens, this.config["max_width"]);

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

export class TokenProcessor extends Processor {
  constructor(config: any, tokenizer: any, wordsSplitter: WhitespaceTokenSplitter) {
    super(config, tokenizer, wordsSplitter);
  }

  prepareBatch(texts: string[], entities: string[]): Record<string, any> {
    const [batchTokens, batchWordsStartIdx, batchWordsEndIdx]: [
      string[][],
      number[][],
      number[][],
    ] = this.batchTokenizeText(texts);
    const { idToClass }: { idToClass: Record<number, string> } = this.createMappings(entities);
    const [inputTokens, textLengths, promptLengths]: [string[][], number[], number[]] =
      this.prepareTextInputs(batchTokens, entities);

    let [inputsIds, attentionMasks, wordsMasks]: [number[][], number[][], number[][]] =
      this.encodeInputs(inputTokens, promptLengths);

    inputsIds = this.padArray(inputsIds);
    attentionMasks = this.padArray(attentionMasks);
    wordsMasks = this.padArray(wordsMasks);

    return {
      inputsIds: inputsIds,
      attentionMasks: attentionMasks,
      wordsMasks: wordsMasks,
      textLengths: textLengths,
      idToClass: idToClass,
      batchTokens: batchTokens,
      batchWordsStartIdx: batchWordsStartIdx,
      batchWordsEndIdx: batchWordsEndIdx,
    };
  }
}
