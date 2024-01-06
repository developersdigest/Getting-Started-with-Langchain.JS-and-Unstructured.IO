// 1. Import dependencies
import axios from "axios";
import { UnstructuredLoader } from "langchain/document_loaders/fs/unstructured";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "@langchain/core/documents";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
// 2. Load environment variables
dotenv.config();
// 3. Configuration object
const config = {
  domain: "https://news.ycombinator.com",
  query: "What is the second story on hacker news?",
};
// 4. Initialize global variables
let currentStep = 1;
const startTime = performance.now();
// 5. Helper function to log messages with elapsed time and write out response
function logTimeWriteOutStep(message, response = null) {
  const elapsedTime = `[${((performance.now() - startTime) / 1000).toFixed(2)}s]`;
  const logMessage = `${elapsedTime} Step ${currentStep}: ${message}`;
  console.log(logMessage);
  response ? console.log(response) : null;
  currentStep += 1;
}
async function main() {
// 6. Starting the main function
  logTimeWriteOutStep("Starting main function");
// 7. Initialize OpenAI Embeddings
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    batchSize: 512,
  });
  logTimeWriteOutStep("OpenAI Embeddings initialized");
// 8. Perform a GET request to the specified domain
  const response = await axios.get(config.domain);
  logTimeWriteOutStep("GET request to Hacker News completed");
// 9. Check and create cache directory
  const cacheDir = path.join(__dirname, "cache");
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir);
  }
  logTimeWriteOutStep("Cache directory checked/created");
// 10. Save response data as an HTML file
  const filePath = path.join(cacheDir, "response.html");
  fs.writeFileSync(filePath, response.data);
  logTimeWriteOutStep("Response data saved as HTML file");
// 11. Load HTML file using UnstructuredLoader
  const loader = new UnstructuredLoader(filePath, { apiKey: process.env.UNSTRUCTURED_API_KEY });
  const loadedData = await loader.load();
  logTimeWriteOutStep("HTML file loaded");
// 12. Convert loaded data into Document objects
  const docs = loadedData.map((item) => new Document({ pageContent: item.pageContent, metadata: item.metadata }));
  logTimeWriteOutStep("Loaded data converted into Document objects");
// 13. Generate embeddings and create a vector store
  const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);
  logTimeWriteOutStep("Vector store created with embeddings");
// 14. Set up QA Chain
  const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    returnSourceDocuments: true,
  });
  logTimeWriteOutStep("QA Chain set up");
// 15. Run QA Chain
  const resp = await chain.call({ query: config.query });
  logTimeWriteOutStep("QA Response", resp.text);
// 16. End timing the execution
  logTimeWriteOutStep("Execution timing ended");
}
// 17. Run Main Function
main();
