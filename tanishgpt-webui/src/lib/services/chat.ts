import { activeSessionId } from "$lib/stores/session";

export async function sendToLlama(message: string, sessionId: string | null) {
    const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            message,
            top_n: 5
        })
    });

    const data = await res.json();
    activeSessionId.set(data.session_id);
    return data.reply;
}
