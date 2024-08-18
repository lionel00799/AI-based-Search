// This file is auto-generated by @hey-api/openapi-ts

export const $AgentFinishStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "agent-finish",
    },
  },
  type: "object",
  title: "AgentFinishStream",
} as const;

export const $AgentQueryPlanStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "agent-query-plan",
    },
    steps: {
      items: {
        type: "string",
      },
      type: "array",
      title: "Steps",
    },
  },
  type: "object",
  title: "AgentQueryPlanStream",
} as const;

export const $AgentReadResultsStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "agent-read-results",
    },
    step_number: {
      type: "integer",
      title: "Step Number",
    },
    results: {
      items: {
        $ref: "#/components/schemas/SearchResult",
      },
      type: "array",
      title: "Results",
    },
  },
  type: "object",
  required: ["step_number"],
  title: "AgentReadResultsStream",
} as const;

export const $AgentSearchFullResponse = {
  properties: {
    steps: {
      items: {
        type: "string",
      },
      type: "array",
      title: "Steps",
    },
    steps_details: {
      items: {
        $ref: "#/components/schemas/AgentSearchStep",
      },
      type: "array",
      title: "Steps Details",
    },
  },
  type: "object",
  title: "AgentSearchFullResponse",
} as const;

export const $AgentSearchFullResponseStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "agent-full-response",
    },
    response: {
      $ref: "#/components/schemas/AgentSearchFullResponse",
    },
  },
  type: "object",
  required: ["response"],
  title: "AgentSearchFullResponseStream",
} as const;

export const $AgentSearchQueriesStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "agent-search-queries",
    },
    step_number: {
      type: "integer",
      title: "Step Number",
    },
    queries: {
      items: {
        type: "string",
      },
      type: "array",
      title: "Queries",
    },
  },
  type: "object",
  required: ["step_number"],
  title: "AgentSearchQueriesStream",
} as const;

export const $AgentSearchStep = {
  properties: {
    step_number: {
      type: "integer",
      title: "Step Number",
    },
    step: {
      type: "string",
      title: "Step",
    },
    queries: {
      items: {
        type: "string",
      },
      type: "array",
      title: "Queries",
    },
    results: {
      items: {
        $ref: "#/components/schemas/SearchResult",
      },
      type: "array",
      title: "Results",
    },
    status: {
      allOf: [
        {
          $ref: "#/components/schemas/AgentSearchStepStatus",
        },
      ],
      default: "default",
    },
  },
  type: "object",
  required: ["step_number", "step"],
  title: "AgentSearchStep",
} as const;

export const $AgentSearchStepStatus = {
  type: "string",
  enum: ["done", "current", "default"],
  title: "AgentSearchStepStatus",
} as const;

export const $BeginStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "begin-stream",
    },
    query: {
      type: "string",
      title: "Query",
    },
  },
  type: "object",
  required: ["query"],
  title: "BeginStream",
} as const;

export const $ChatHistoryResponse = {
  properties: {
    snapshots: {
      items: {
        $ref: "#/components/schemas/ChatSnapshot",
      },
      type: "array",
      title: "Snapshots",
    },
  },
  type: "object",
  title: "ChatHistoryResponse",
} as const;

export const $ChatMessage = {
  properties: {
    content: {
      type: "string",
      title: "Content",
    },
    role: {
      $ref: "#/components/schemas/MessageRole",
    },
    related_queries: {
      anyOf: [
        {
          items: {
            type: "string",
          },
          type: "array",
        },
        {
          type: "null",
        },
      ],
      title: "Related Queries",
    },
    sources: {
      anyOf: [
        {
          items: {
            $ref: "#/components/schemas/SearchResult",
          },
          type: "array",
        },
        {
          type: "null",
        },
      ],
      title: "Sources",
    },
    images: {
      anyOf: [
        {
          items: {
            type: "string",
          },
          type: "array",
        },
        {
          type: "null",
        },
      ],
      title: "Images",
    },
    is_error_message: {
      type: "boolean",
      title: "Is Error Message",
      default: false,
    },
    agent_response: {
      anyOf: [
        {
          $ref: "#/components/schemas/AgentSearchFullResponse",
        },
        {
          type: "null",
        },
      ],
    },
  },
  type: "object",
  required: ["content", "role"],
  title: "ChatMessage",
} as const;

export const $ChatModel = {
  type: "string",
  enum: [
    "llama-3-70b",
    "gpt-4o",
    "gpt-3.5-turbo",
    "llama3.1:8b-instruct-q4_0",
    "gemma",
    "mistral",
    "phi3:14b",
    "custom",
  ],
  title: "ChatModel",
} as const;

export const $ChatRequest = {
  properties: {
    thread_id: {
      anyOf: [
        {
          type: "integer",
        },
        {
          type: "null",
        },
      ],
      title: "Thread Id",
    },
    query: {
      type: "string",
      title: "Query",
    },
    history: {
      items: {
        $ref: "#/components/schemas/Message",
      },
      type: "array",
      title: "History",
    },
    model: {
      allOf: [
        {
          $ref: "#/components/schemas/ChatModel",
        },
      ],
      default: "gpt-3.5-turbo",
    },
    pro_search: {
      type: "boolean",
      title: "Pro Search",
      default: false,
    },
  },
  type: "object",
  required: ["query"],
  title: "ChatRequest",
} as const;

export const $ChatResponseEvent = {
  properties: {
    event: {
      $ref: "#/components/schemas/StreamEvent",
    },
    data: {
      anyOf: [
        {
          $ref: "#/components/schemas/BeginStream",
        },
        {
          $ref: "#/components/schemas/SearchResultStream",
        },
        {
          $ref: "#/components/schemas/TextChunkStream",
        },
        {
          $ref: "#/components/schemas/RelatedQueriesStream",
        },
        {
          $ref: "#/components/schemas/StreamEndStream",
        },
        {
          $ref: "#/components/schemas/FinalResponseStream",
        },
        {
          $ref: "#/components/schemas/ErrorStream",
        },
        {
          $ref: "#/components/schemas/AgentQueryPlanStream",
        },
        {
          $ref: "#/components/schemas/AgentSearchQueriesStream",
        },
        {
          $ref: "#/components/schemas/AgentReadResultsStream",
        },
        {
          $ref: "#/components/schemas/AgentFinishStream",
        },
        {
          $ref: "#/components/schemas/AgentSearchFullResponseStream",
        },
      ],
      title: "Data",
    },
  },
  type: "object",
  required: ["event", "data"],
  title: "ChatResponseEvent",
} as const;

export const $ChatSnapshot = {
  properties: {
    id: {
      type: "integer",
      title: "Id",
    },
    title: {
      type: "string",
      title: "Title",
    },
    date: {
      type: "string",
      format: "date-time",
      title: "Date",
    },
    preview: {
      type: "string",
      title: "Preview",
    },
    modelName: {
      type: "string",
      title: "Model Name",
    },
  },
  type: "object",
  required: ["id", "title", "date", "preview", "modelName"],
  title: "ChatSnapshot",
} as const;

export const $ErrorStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "error",
    },
    detail: {
      type: "string",
      title: "Detail",
    },
  },
  type: "object",
  required: ["detail"],
  title: "ErrorStream",
} as const;

export const $FinalResponseStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "final-response",
    },
    message: {
      type: "string",
      title: "Message",
    },
  },
  type: "object",
  required: ["message"],
  title: "FinalResponseStream",
} as const;

export const $HTTPValidationError = {
  properties: {
    detail: {
      items: {
        $ref: "#/components/schemas/ValidationError",
      },
      type: "array",
      title: "Detail",
    },
  },
  type: "object",
  title: "HTTPValidationError",
} as const;

export const $Message = {
  properties: {
    content: {
      type: "string",
      title: "Content",
    },
    role: {
      $ref: "#/components/schemas/MessageRole",
    },
  },
  type: "object",
  required: ["content", "role"],
  title: "Message",
} as const;

export const $MessageRole = {
  type: "string",
  enum: ["user", "assistant"],
  title: "MessageRole",
} as const;

export const $RelatedQueriesStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "related-queries",
    },
    related_queries: {
      items: {
        type: "string",
      },
      type: "array",
      title: "Related Queries",
    },
  },
  type: "object",
  title: "RelatedQueriesStream",
} as const;

export const $SearchResult = {
  properties: {
    title: {
      type: "string",
      title: "Title",
    },
    url: {
      type: "string",
      title: "Url",
    },
    content: {
      type: "string",
      title: "Content",
    },
  },
  type: "object",
  required: ["title", "url", "content"],
  title: "SearchResult",
} as const;

export const $SearchResultStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "search-results",
    },
    results: {
      items: {
        $ref: "#/components/schemas/SearchResult",
      },
      type: "array",
      title: "Results",
    },
    images: {
      items: {
        type: "string",
      },
      type: "array",
      title: "Images",
    },
  },
  type: "object",
  title: "SearchResultStream",
} as const;

export const $StreamEndStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "stream-end",
    },
    thread_id: {
      anyOf: [
        {
          type: "integer",
        },
        {
          type: "null",
        },
      ],
      title: "Thread Id",
    },
  },
  type: "object",
  title: "StreamEndStream",
} as const;

export const $StreamEvent = {
  type: "string",
  enum: [
    "begin-stream",
    "search-results",
    "text-chunk",
    "related-queries",
    "stream-end",
    "final-response",
    "error",
    "agent-query-plan",
    "agent-search-queries",
    "agent-read-results",
    "agent-finish",
    "agent-full-response",
  ],
  title: "StreamEvent",
} as const;

export const $TextChunkStream = {
  properties: {
    event_type: {
      allOf: [
        {
          $ref: "#/components/schemas/StreamEvent",
        },
      ],
      default: "text-chunk",
    },
    text: {
      type: "string",
      title: "Text",
    },
  },
  type: "object",
  required: ["text"],
  title: "TextChunkStream",
} as const;

export const $ThreadResponse = {
  properties: {
    thread_id: {
      type: "integer",
      title: "Thread Id",
    },
    messages: {
      items: {
        $ref: "#/components/schemas/ChatMessage",
      },
      type: "array",
      title: "Messages",
    },
  },
  type: "object",
  required: ["thread_id"],
  title: "ThreadResponse",
} as const;

export const $ValidationError = {
  properties: {
    loc: {
      items: {
        anyOf: [
          {
            type: "string",
          },
          {
            type: "integer",
          },
        ],
      },
      type: "array",
      title: "Location",
    },
    msg: {
      type: "string",
      title: "Message",
    },
    type: {
      type: "string",
      title: "Error Type",
    },
  },
  type: "object",
  required: ["loc", "msg", "type"],
  title: "ValidationError",
} as const;