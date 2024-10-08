import ort from "onnxruntime-web";
import { ONNXWrapper } from "./ONNXWrapper";
import { RawInferenceResult } from "./Gliner";

export class Model {
  constructor(
    public config: any,
    public processor: any,
    public decoder: any,
    public onnxWrapper: ONNXWrapper,
  ) {}

  async initialize(): Promise<void> {
    await this.onnxWrapper.init();
  }
}

export class SpanModel extends Model {
  prepareInputs(batch: any): Record<string, ort.Tensor> {
    const batch_size: number = batch.inputsIds.length;
    const num_tokens: number = batch.inputsIds[0].length;
    const num_spans: number = batch.spanIdxs[0].length;

    const createTensor = (data: any[], shape: number[], tensorType: any = "int64"): ort.Tensor => {
      // @ts-ignore // NOTE: node types not working
      return new this.onnxWrapper.ort.Tensor(tensorType, data.flat(Infinity), shape);
    };

    let input_ids: ort.Tensor = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask: ort.Tensor = createTensor(batch.attentionMasks, [batch_size, num_tokens]); // NOTE: why convert to bool but type is not bool?
    let words_mask: ort.Tensor = createTensor(batch.wordsMasks, [batch_size, num_tokens]);
    let text_lengths: ort.Tensor = createTensor(batch.textLengths, [batch_size, 1]);
    let span_idx: ort.Tensor = createTensor(batch.spanIdxs, [batch_size, num_spans, 2]);
    let span_mask: ort.Tensor = createTensor(batch.spanMasks, [batch_size, num_spans], "bool");

    const feeds: Record<string, ort.Tensor> = {
      input_ids: input_ids,
      attention_mask: attention_mask,
      words_mask: words_mask,
      text_lengths: text_lengths,
      span_idx: span_idx,
      span_mask: span_mask,
    };

    return feeds;
  }

  async inference(
    texts: string[],
    entities: string[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
  ): Promise<RawInferenceResult> {
    let batch: Record<string, any> = this.processor.prepareBatch(texts, entities);
    let feeds: Record<string, ort.Tensor> = this.prepareInputs(batch);
    const results: Record<string, ort.Tensor> = await this.onnxWrapper.run(feeds);
    const modelOutput: any = results["logits"].data;
    // const modelOutput = results.logits.data as number[];

    const batchSize: number = batch.batchTokens.length;
    const inputLength: number = Math.max(...batch.textLengths);
    const maxWidth: number = this.config.max_width;
    const numEntities: number = entities.length;
    const batchIds: number[] = Array.from({ length: batchSize }, (_, i) => i);
    const decodedSpans: RawInferenceResult = this.decoder.decode(
      batchSize,
      inputLength,
      maxWidth,
      numEntities,
      texts,
      batchIds,
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      modelOutput,
      flatNer,
      threshold,
      multiLabel,
    );

    return decodedSpans;
  }

  async inference_with_chunking(
    texts: string[],
    entities: string[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
    batch_size: number = 4,
    max_words: number = 512,
  ): Promise<RawInferenceResult> {
    const {
      idToClass,
    }: {
      idToClass: Record<number, string>;
    } = this.processor.createMappings(entities);
    let batchIds: number[] = [];
    let batchTokens: string[][] = [];
    let batchWordsStartIdx: number[][] = [];
    let batchWordsEndIdx: number[][] = [];
    texts.forEach((text, id) => {
      let [tokens, wordsStartIdx, wordsEndIdx]: [string[], number[], number[]] =
        this.processor.tokenizeText(text);
      let num_sub_batches: number = Math.ceil(tokens.length / max_words);

      for (let i = 0; i < num_sub_batches; i++) {
        let start: number = i * max_words;
        let end: number = Math.min((i + 1) * max_words, tokens.length);

        batchIds.push(id);
        batchTokens.push(tokens.slice(start, end));
        batchWordsStartIdx.push(wordsStartIdx.slice(start, end));
        batchWordsEndIdx.push(wordsEndIdx.slice(start, end));
      }
    });

    let num_batches: number = Math.ceil(batchIds.length / batch_size);

    let finalDecodedSpans: RawInferenceResult = [];
    for (let id = 0; id < texts.length; id++) {
      finalDecodedSpans.push([]);
    }

    for (let batch_id = 0; batch_id < num_batches; batch_id++) {
      let start: number = batch_id * batch_size;
      let end: number = Math.min((batch_id + 1) * batch_size, batchIds.length);

      let currBatchTokens: string[][] = batchTokens.slice(start, end);
      let currBatchWordsStartIdx: number[][] = batchWordsStartIdx.slice(start, end);
      let currBatchWordsEndIdx: number[][] = batchWordsEndIdx.slice(start, end);
      let currBatchIds: number[] = batchIds.slice(start, end);

      let [inputTokens, textLengths, promptLengths]: [string[][], number[], number[]] =
        this.processor.prepareTextInputs(currBatchTokens, entities);

      let [inputsIds, attentionMasks, wordsMasks]: [number[][], number[][], number[][]] =
        this.processor.encodeInputs(inputTokens, promptLengths);

      inputsIds = this.processor.padArray(inputsIds);
      attentionMasks = this.processor.padArray(attentionMasks);
      wordsMasks = this.processor.padArray(wordsMasks);

      let { spanIdxs, spanMasks }: { spanIdxs: number[][][]; spanMasks: boolean[][] } =
        this.processor.prepareSpans(currBatchTokens, this.config["max_width"]);

      spanIdxs = this.processor.padArray(spanIdxs, 3);
      spanMasks = this.processor.padArray(spanMasks);

      let batch: Record<string, any> = {
        inputsIds: inputsIds,
        attentionMasks: attentionMasks,
        wordsMasks: wordsMasks,
        textLengths: textLengths,
        spanIdxs: spanIdxs,
        spanMasks: spanMasks,
        idToClass: idToClass,
        batchTokens: currBatchTokens,
        batchWordsStartIdx: currBatchWordsStartIdx,
        batchWordsEndIdx: currBatchWordsEndIdx,
      };

      let feeds: Record<string, ort.Tensor> = this.prepareInputs(batch);
      const results: Record<string, ort.Tensor> = await this.onnxWrapper.run(feeds);
      const modelOutput: number[] = results["logits"].data;
      // const modelOutput = results.logits.data as number[];

      const batchSize: number = batch.batchTokens.length;
      const inputLength: number = Math.max(...batch.textLengths);
      const maxWidth: number = this.config.max_width;
      const numEntities: number = entities.length;

      const decodedSpans: RawInferenceResult = this.decoder.decode(
        batchSize,
        inputLength,
        maxWidth,
        numEntities,
        texts,
        currBatchIds,
        batch.batchWordsStartIdx,
        batch.batchWordsEndIdx,
        batch.idToClass,
        modelOutput,
        flatNer,
        threshold,
        multiLabel,
      );

      for (let i = 0; i < currBatchIds.length; i++) {
        const originalTextId = currBatchIds[i];
        finalDecodedSpans[originalTextId].push(...decodedSpans[i]);
      }
    }

    return finalDecodedSpans;
  }
}

export class TokenModel extends Model {
  prepareInputs(batch: any): Record<string, ort.Tensor> {
    const batch_size: number = batch.inputsIds.length;
    const num_tokens: number = batch.inputsIds[0].length;

    const createTensor = (data: any[], shape: number[], tensorType: any = "int64"): ort.Tensor => {
      // @ts-ignore // NOTE: node types not working
      return new this.onnxWrapper.ort.Tensor(tensorType, data.flat(Infinity), shape);
    };

    let input_ids: ort.Tensor = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask: ort.Tensor = createTensor(batch.attentionMasks, [batch_size, num_tokens]); // NOTE: why convert to bool but type is not bool?
    let words_mask: ort.Tensor = createTensor(batch.wordsMasks, [batch_size, num_tokens]);
    let text_lengths: ort.Tensor = createTensor(batch.textLengths, [batch_size, 1]);

    const feeds: Record<string, ort.Tensor> = {
      input_ids: input_ids,
      attention_mask: attention_mask,
      words_mask: words_mask,
      text_lengths: text_lengths,
    };

    return feeds;
  }

  async inference(
    texts: string[],
    entities: string[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
  ): Promise<RawInferenceResult> {
    let batch: Record<string, any> = this.processor.prepareBatch(texts, entities);
    let feeds: Record<string, ort.Tensor> = this.prepareInputs(batch);
    const results: Record<string, ort.Tensor> = await this.onnxWrapper.run(feeds);
    const modelOutput: any = results["logits"].data;
    // const modelOutput = results.logits.data as number[];

    const batchSize: number = batch.batchTokens.length;
    const inputLength: number = Math.max(...batch.textLengths);
    const numEntities: number = entities.length;
    const batchIds: number[] = Array.from({ length: batchSize }, (_, i) => i);
    const decodedSpans: RawInferenceResult = this.decoder.decode(
      batchSize,
      inputLength,
      numEntities,
      texts,
      batchIds,
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      modelOutput,
      flatNer,
      threshold,
      multiLabel,
    );

    return decodedSpans;
  }
  async inference_with_chunking(
    texts: string[],
    entities: string[],
    flatNer: boolean = true,
    threshold: number = 0.5,
    multiLabel: boolean = false,
    batch_size: number = 4,
    max_words: number = 512,
  ): Promise<RawInferenceResult> {
    const {
      idToClass,
    }: {
      idToClass: Record<number, string>;
    } = this.processor.createMappings(entities);
    let batchIds: number[] = [];
    let batchTokens: string[][] = [];
    let batchWordsStartIdx: number[][] = [];
    let batchWordsEndIdx: number[][] = [];
    texts.forEach((text, id) => {
      let [tokens, wordsStartIdx, wordsEndIdx]: [string[], number[], number[]] =
        this.processor.tokenizeText(text);
      let num_sub_batches: number = Math.ceil(tokens.length / max_words);

      for (let i = 0; i < num_sub_batches; i++) {
        let start: number = i * max_words;
        let end: number = Math.min((i + 1) * max_words, tokens.length);

        batchIds.push(id);
        batchTokens.push(tokens.slice(start, end));
        batchWordsStartIdx.push(wordsStartIdx.slice(start, end));
        batchWordsEndIdx.push(wordsEndIdx.slice(start, end));
      }
    });

    let num_batches: number = Math.ceil(batchIds.length / batch_size);

    let finalDecodedSpans: RawInferenceResult = [];
    for (let id = 0; id < texts.length; id++) {
      finalDecodedSpans.push([]);
    }

    for (let batch_id = 0; batch_id < num_batches; batch_id++) {
      let start: number = batch_id * batch_size;
      let end: number = Math.min((batch_id + 1) * batch_size, batchIds.length);

      let currBatchTokens: string[][] = batchTokens.slice(start, end);
      let currBatchWordsStartIdx: number[][] = batchWordsStartIdx.slice(start, end);
      let currBatchWordsEndIdx: number[][] = batchWordsEndIdx.slice(start, end);
      let currBatchIds: number[] = batchIds.slice(start, end);

      let [inputTokens, textLengths, promptLengths]: [string[][], number[], number[]] =
        this.processor.prepareTextInputs(currBatchTokens, entities);

      let [inputsIds, attentionMasks, wordsMasks]: [number[][], number[][], number[][]] =
        this.processor.encodeInputs(inputTokens, promptLengths);

      inputsIds = this.processor.padArray(inputsIds);
      attentionMasks = this.processor.padArray(attentionMasks);
      wordsMasks = this.processor.padArray(wordsMasks);

      let batch: Record<string, any> = {
        inputsIds: inputsIds,
        attentionMasks: attentionMasks,
        wordsMasks: wordsMasks,
        textLengths: textLengths,
        idToClass: idToClass,
        batchTokens: currBatchTokens,
        batchWordsStartIdx: currBatchWordsStartIdx,
        batchWordsEndIdx: currBatchWordsEndIdx,
      };

      let feeds: Record<string, ort.Tensor> = this.prepareInputs(batch);
      const results: Record<string, ort.Tensor> = await this.onnxWrapper.run(feeds);
      const modelOutput: number[] = results["logits"].data;
      // const modelOutput = results.logits.data as number[];

      const batchSize: number = batch.batchTokens.length;
      const inputLength: number = Math.max(...batch.textLengths);
      const numEntities: number = entities.length;

      const decodedSpans: RawInferenceResult = this.decoder.decode(
        batchSize,
        inputLength,
        numEntities,
        texts,
        currBatchIds,
        batch.batchWordsStartIdx,
        batch.batchWordsEndIdx,
        batch.idToClass,
        modelOutput,
        flatNer,
        threshold,
        multiLabel,
      );

      for (let i = 0; i < currBatchIds.length; i++) {
        const originalTextId = currBatchIds[i];
        finalDecodedSpans[originalTextId].push(...decodedSpans[i]);
      }
    }

    return finalDecodedSpans;
  }
}
