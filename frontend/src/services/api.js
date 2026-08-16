const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Search and get answer from the legal AI backend
 * @param {string} question - The user's question
 * @param {number} topK - Number of top results to return (default: 8)
 * @param {boolean} useLLM - Whether to use LLM for answer generation (default: true)
 * @returns {Promise<{answer: string, sources: string}>}
 */
export async function searchAndAnswer(question, topK = 8, useLLM = true) {
  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: topK,
        use_llm: useLLM,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return {
      answer: data.answer,
      sources: data.sources,
    };
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

/**
 * Health check for the API
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
