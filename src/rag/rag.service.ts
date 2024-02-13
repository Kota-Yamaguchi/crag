import { Injectable } from '@nestjs/common';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { ToolExecutor } from '@langchain/langgraph/prebuilt';
import { convertToOpenAIFunction } from '@langchain/core/utils/function_calling';
import { BaseMessageChunk } from '@langchain/core/messages';
import { StateGraph, END } from '@langchain/langgraph';
import { Runnable, RunnableLambda } from '@langchain/core/runnables';
import {
  ChatOpenAICallOptions,
  OpenAIEmbeddings,
  ChatOpenAI,
} from '@langchain/openai';
import { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { loadEvaluator } from 'langchain/evaluation';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { PromptTemplate } from '@langchain/core/prompts';
import { PlaywrightWebBaseLoader } from 'langchain/document_loaders/web/playwright';
import { Document } from '@langchain/core/documents';
import { formatDocumentsAsString } from 'langchain/util/document';
import { LangChainTracer } from 'langchain/callbacks';
import { Client } from 'langsmith';
@Injectable()
export class RagService {
  private toolExecutor: ToolExecutor;
  private model: Runnable<
    BaseLanguageModelInput,
    BaseMessageChunk,
    ChatOpenAICallOptions
  >;
  private embeddings: OpenAIEmbeddings;
  private tracer: LangChainTracer;
  constructor() {
    //ツールを定義する。
    //ここでは、TavilySearchResultsを使用しています
    const tools = [
      new TavilySearchResults({
        maxResults: 1,
        apiKey: process.env.TAVILY_API_KEY,
      }),
    ];
    this.toolExecutor = new ToolExecutor({
      tools,
    });
    //modelを定義する。
    //ここでは、ChatOpenAIを使用しています
    const model = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
      temperature: 0,
      streaming: true,
    });
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
    //toolsをFunction Callingに変換し、modelにバインドする
    // これをすることで、modelはtoolsを呼び出すことができる
    const toolsAsOpenAIFunctions = tools.map((tool) =>
      convertToOpenAIFunction(tool),
    );
    this.model = model.bind({
      functions: toolsAsOpenAIFunctions,
    });
    const client = new Client({
      apiUrl: process.env.LANGCHAIN_ENDPOINT,
      apiKey: process.env.LANGCHAIN_API_KEY,
    });
    this.tracer = new LangChainTracer({
      projectName: process.env.LANGCHAIN_PROJECT,
      client,
    });
  }

  /**
   *
   * @returns
   * @description
   * この関数は、Cragを実行します。
   */
  async executeCrag(): Promise<string> {
    console.log('Executing Crag');
    const retriever = await this.setupRetriever();
    type Keys = {
      question: string;
      documents: Document[];
      [key: string]: any;
    };
    const agentState = {
      keys: {
        value: (_, keys: Keys) => {
          return keys;
        },
        default: () => ({ question: '', documents: [] }),
      },
    };

    // 各ノードで実行する関数を定義します。
    const retrieveDocuments = async (state: { keys: Keys }) => {
      console.log('retrieve documents');
      const { question } = state.keys;

      const retrieverResult = await retriever.getRelevantDocuments(question);
      const response = {
        keys: {
          question: question,
          documents: retrieverResult,
        },
      };
      return response;
    };

    const gradeDocuments = async (state: { keys: Keys }) => {
      console.log('grade documents');
      const { question, documents } = state.keys;
      // 評価基準を設定します。ここでは、'relevant', 'irrelevant', 'ambiguous' の3つのカテゴリを持つ例を示します。

      // Criteria Evaluatorをロードします。
      const evaluator = await loadEvaluator('criteria', {
        criteria: 'relevance',
      });

      // 文書とクエリを使用して評価を実行します。
      //documentはstate.messagesから取得する.0番目以外のメッセージを取得する
      const evalResult = (await evaluator.evaluateStrings(
        {
          input: question, // クエリ
          prediction: formatDocumentsAsString(documents), // 評価する文書
        },
        { callbacks: [this.tracer] },
      )) as {
        reasoning: string;
        value: 'Y' | 'N';
        score: number;
      };
      console.log('eval', evalResult.value, evalResult.reasoning);

      let search = false;
      if (evalResult.value === 'N') {
        search = true;
      }
      return {
        keys: {
          documents: documents,
          question: question,
          run_web_search: search,
        },
      };
    };

    const decideToGenerate = (state: { keys: Keys }) => {
      const { run_web_search } = state.keys;
      // If there is no function call, then we finish
      if (!run_web_search) {
        console.log('decide to generate');
        return 'generate';
      }
      console.log('decide to transform query');
      return 'transform_query';
    };

    const transformQuery = async (state: { keys: Keys }) => {
      const { question, documents } = state.keys;
      const chain = PromptTemplate.fromTemplate(
        `You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: `,
      ).pipe(this.model);

      const response = await chain.invoke(
        { question: question },
        { callbacks: [this.tracer] },
      );
      console.log('transform query:', response);
      return { keys: { documents: documents, question: response.content } };
    };

    const webSearch = async (state: { keys: Keys }) => {
      const { question, documents } = state.keys;
      console.log('web search', question);
      const response = (await this.toolExecutor.invoke(
        {
          tool: 'tavily_search_results_json',
          toolInput: question,
          log: '',
        },
        { callbacks: [this.tracer] },
      )) as string;
      //responseをパースして、Documentに変換する
      const parsedResponse = JSON.parse(response);
      const webResults = parsedResponse
        .map((d: any) => d['content'])
        .join('\n');
      const webResultsDocument = new Document({ pageContent: webResults });
      documents.push(webResultsDocument);
      return { keys: { documents: documents, question: question } };
    };

    // Define the function that calls the model
    const generateAnswer = async (state: { keys: Keys }) => {
      console.log('generate answer');
      const { question, documents } = state.keys;
      const prompt = PromptTemplate.fromTemplate(
        `
        Please answer the following questions according to the instructions on the following information.
        {context}
        Question: {question}
       `,
      ).pipe(this.model);
      const response = await prompt.invoke(
        {
          context: formatDocumentsAsString(documents),
          question: question,
        },
        { callbacks: [this.tracer] },
      );
      // We return a list, because this will get added to the existing list
      return {
        keys: {
          documents: documents,
          question: question,
          generation: response.content,
        },
      };
    };

    console.log('Creating workflow');
    // Define a new graph
    const workflow = new StateGraph({
      channels: agentState,
    });

    workflow.addNode(
      'retrieve',
      new RunnableLambda({ func: retrieveDocuments }),
    );
    workflow.addNode(
      'grade_documents',
      new RunnableLambda({ func: gradeDocuments }),
    );

    workflow.addNode(
      'transform_query',
      new RunnableLambda({ func: transformQuery }),
    );
    workflow.addNode('web_search', new RunnableLambda({ func: webSearch }));
    workflow.addNode('generate', new RunnableLambda({ func: generateAnswer }));

    // 条件付きエッジを追加します。この例では、'grade_documents' ノードから 'generate' または 'transform_query' への分岐を行います。判定は `decideToGenerate` によって行われます。
    workflow.addConditionalEdges('grade_documents', decideToGenerate, {
      generate: 'generate',
      transform_query: 'transform_query',
    });

    // 通常のエッジを追加します。'retrieve' から 'grade_documents' へ進みます。
    workflow.addEdge('retrieve', 'grade_documents');
    //'transform_query' から 'web_search' へ、そして 'web_search' から 'generate' へと進みます。
    workflow.addEdge('transform_query', 'web_search');
    workflow.addEdge('web_search', 'generate');
    // 最後に、generateで処理は終了します。
    workflow.addEdge('generate', END);
    // エントリポイントを設定します。
    workflow.setEntryPoint('retrieve');
    const app = workflow.compile();
    const inputs = {
      keys: {
        question: 'Explain how the different types of agent memory work?',
        documents: [],
      },
    };
    console.log('Invoking workflow');
    const result = (await app.invoke(inputs)) as { keys: Keys };
    console.log('Workflow complete');
    console.log('Result', result);
    return result.keys.generation;
  }

  private async setupRetriever() {
    console.log('Setting up retriever');
    const urls = [
      'https://lilianweng.github.io/posts/2023-06-23-agent/',
      'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
      'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/',
    ];
    const docsPromises = urls.map((url) =>
      new PlaywrightWebBaseLoader(url).load(),
    );
    const docs = await Promise.all(docsPromises);
    const docsList: string[] = ([] as string[]).concat(
      ...docs.map((docArray) => docArray.map((doc) => doc.pageContent)),
    );

    // Split documents
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 3000,
      chunkOverlap: 300,
    });

    const docSplits = await splitter.createDocuments(docsList);
    const store = await MemoryVectorStore.fromDocuments(
      docSplits,
      this.embeddings,
    );
    const retriever = store.asRetriever();
    console.log('Retriever setup complete');
    return retriever;
  }
}
