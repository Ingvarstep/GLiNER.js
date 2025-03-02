import { AutoTokenizer, env } from "@xenova/transformers";
import { SpanModel, TokenModel } from "../lib/model";
import { WhitespaceTokenSplitter, SpanProcessor, TokenProcessor } from "../lib/processor";
import { SpanDecoder, TokenDecoder } from "../lib/decoder";
import { ONNXNodeWrapper } from "./ONNXNodeWrapper";
import {
  IEntityResult,
  IInference,
  InferenceResultMultiple,
  InitConfig,
  RawInferenceResult,
} from "../interfaces";

export class Gliner {
  private model: SpanModel | TokenModel | null = null;

  constructor(private config: InitConfig) {
    env.allowLocalModels = config.transformersSettings?.allowLocalModels ?? false;
    env.useBrowserCache = config.transformersSettings?.useBrowserCache ?? false;

    this.config = {
      ...config,
      maxWidth: config.maxWidth || 12,
      modelType: config.modelType || "span-level",
    };
  }

  async initialize(): Promise<void> {
    const { tokenizerPath, onnxSettings, maxWidth } = this.config;

    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
    // console.log("Tokenizer loaded.");

    const wordSplitter = new WhitespaceTokenSplitter();

    const onnxWrapper = new ONNXNodeWrapper(onnxSettings);

    if (this.config.modelType == "span-level") {
      // console.log("Initializing Span-level Model...");
      const processor = new SpanProcessor({ max_width: maxWidth }, tokenizer, wordSplitter);
      const decoder = new SpanDecoder({ max_width: maxWidth });

      this.model = new SpanModel({ max_width: maxWidth }, processor, decoder, onnxWrapper);
    } else {
      // console.log("Initializing Token-level Model...");

      const processor = new TokenProcessor({ max_width: maxWidth }, tokenizer, wordSplitter);
      const decoder = new TokenDecoder({ max_width: maxWidth });

      this.model = new TokenModel({ max_width: maxWidth }, processor, decoder, onnxWrapper);
    }
    await this.model.initialize();
  }

  async inference({
    texts,
    entities,
    flatNer = true,
    threshold = 0.5,
    multiLabel = false,
  }: IInference): Promise<InferenceResultMultiple> {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }

    const result = await this.model.inference(texts, entities, flatNer, threshold, multiLabel);
    return this.mapRawResultToResponse(result);
  }

  async inference_with_chunking({
    texts,
    entities,
    flatNer = false,
    threshold = 0.5,
  }: IInference): Promise<InferenceResultMultiple> {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }

    const result = await this.model.inference_with_chunking(texts, entities, flatNer, threshold);
    return this.mapRawResultToResponse(result);
  }

  mapRawResultToResponse(rawResult: RawInferenceResult): InferenceResultMultiple {
    const response: InferenceResultMultiple = [];
    for (const individualResult of rawResult) {
      const entityResult: IEntityResult[] = individualResult.map(
        ([spanText, start, end, label, score]) => ({
          spanText,
          start,
          end,
          label,
          score,
        }),
      );
      response.push(entityResult);
    }

    return response;
  }
}
