const API_BASE = "http://127.0.0.1:8000";

export async function uploadDocument(file: File): Promise<{ doc_id: string; message: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        throw new Error("Upload failed");
    }

    return await res.json();
}
