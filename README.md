This is a GLiNER inference engine written in JavaScript.

Test script:
```JavaScript
import { pipeline, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

import { SpanProcessor, WhitespaceTokenSplitter } from './src/processor.js';

async function main() {
    const config = {"max_width": 12};

    // Load the tokenizer
    const tokenizer = await AutoTokenizer.from_pretrained("onnx-community/gliner_small-v2");
    const wordSplitter = new WhitespaceTokenSplitter();  // No 'await' needed for regular classes
    const processor = new SpanProcessor(config, tokenizer, wordSplitter);  // No 'await' here either

    // Example input text
    const input_text1 = "Kyiv is the capital of Ukraine.";
    const input_text2 = "One day I will see the world!";
    const input_text3 = "Earth is the most beautiful planet in the Universe.";
    
    const texts = [input_text1, input_text2, input_text3]
    const entities = ['city', 'country', 'river', 'person', 'car'];

    let batch = processor.prepareBatch(texts, entities);

    console.log(batch);

    
}

main().catch(error => console.error(error));
```