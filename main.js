import { AutoTokenizer } from '@xenova/transformers';
import { Model } from './src/model.js';
import { WhitespaceTokenSplitter, SpanProcessor} from './src/processor.js';
import { SpanDecoder } from './src/decoder.js';

import * as ort from 'onnxruntime-web';
ort.env.wasm.numThreads =1 


async function main() {
    const config = {"max_width": 12};
    // Load the tokenizer
    const tokenizer = await AutoTokenizer.from_pretrained("onnx-community/gliner_small-v2");
    console.log("Loaded tokenizer.");
    
    const wordSplitter = new WhitespaceTokenSplitter();
    const processor = new SpanProcessor(config, tokenizer, wordSplitter);
    const decoder = new SpanDecoder(config);
    const model = new Model(config, processor, decoder, "model.onnx");
  
    // Initialize the model
    await model.initialize();
  
    // Example input text
    const input_text1 = "Kyiv is the capital of Ukraine.";
    const input_text2 = "One day I will see the world!";
    const input_text3 = "Earth is the most beautiful planet in the Universe.";
    const texts = [input_text1, input_text2, input_text3];
    const entities = ['city', 'country', 'river', 'person', 'car'];
    
    const threshold = 0.3;
    const decoded = await model.inference(texts, entities, false, threshold);
    console.log(decoded);
    document.write(`data of result tensor 'c': ${JSON.stringify(decoded)}`);
  }
  
main().catch(error => console.error(error));