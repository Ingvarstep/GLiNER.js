// import { Gliner } from "../../../src";
import { Gliner } from "../../../dist";
// import { Gliner } from "gliner";

async function main(): Promise<void> {
  const gliner = new Gliner({
    tokenizerPath: "onnx-community/gliner_small-v2",
    onnxSettings: {
      modelPath: "/model.onnx",
    },
    maxWidth: 12,
  });

  await gliner.initialize();

  const input_text1 = `
    Write a white paper on the state of the financial market for Morar - Rice to share with potential investors.
    Please prepare a brief for Gusikowski, Hansen and Shanahan on the legal aspects of piracy and maritime security.
    Write a short article about the differences between mindfulness-based therapy and traditional therapy for Mathias to share on social media.
    Could you write a blog post about the influence of family dynamics on adolescent development for Constantin.Morar's website?
    What's the difference between first-generation and second-generation antipsychotics for residents of 76511?
    Hey, can you help me understand the steps to appeal an administrative law decision in West Virginia?
    4. Write a summary of the key privacy law principles for Optimization businesses to follow.
    `;

  const input_text2 = `
    How can Sadie Turcotte and Clifford Ernser develop better listening skills to improve their marriage?
    Write a white paper on the effectiveness of various ADHD therapy approaches for Trantow Inc to share with their colleagues.
    Could you please create a pricing strategy roadmap for Anastacio to follow over the next year?
    Hey there, can you create a customer satisfaction survey for Flatley, Rohan and Koepp's business? They want to measure their customers' happiness.
    Create a training program for Tonya Quitzon's employees to familiarize them with the business continuity plan.
    HIPAA guidelines for protecting patients' 146.229.205.216 in telemedicine consultations.
    Could you please provide Bruce Buckridge with a list of the top supply chain management software available in the market?
    Can you provide a list of resources for entrepreneurs to learn about intellectual property protection? Send it to Minnie Gulgowski at Kaylie.Littel52@hotmail.com.
    Prepare a trade compliance checklist for Borer LLC to ensure their business is adhering to all relevant trade laws.
    Investigate the effects of academic pressure on adolescent mental health, referencing Bethany Koss's experiences.
    `;

  const texts = [input_text1];
  const entities = ["city", "country", "river", "person", "car"];

  try {
    const start = performance.now();
    console.log("Running inference #1...");
    const decoded = await gliner.inference({ texts, entities });
    console.log(decoded);
    const end = performance.now();
    console.log(`Inference #1 took ${end - start} ms`);

    const start2 = performance.now();
    console.log("Running inference #2...");
    const decoded2 = await gliner.inference({ texts: [input_text2], entities });
    const end2 = performance.now();
    console.log(`Inference #2 took ${end2 - start2} ms`);
    console.log(decoded2);
  } catch (error) {
    console.error("Failed to run inference: ", error);
    throw error;
  }
}

main().catch((error) => console.error(error));
