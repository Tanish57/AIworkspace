// src/lib/services/chat.ts

import { activeSessionId } from "$lib/stores/session";

export async function sendToLlama(message: string, session_id: string | null) {
    const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: session_id,  // backend creates one if null
            message,
            top_n: 5
        })
    });

    const data = await res.json();

    // Store the session ID returned by backend (first run)
    activeSessionId?.set(data.session_id)

    return data.reply;
}
