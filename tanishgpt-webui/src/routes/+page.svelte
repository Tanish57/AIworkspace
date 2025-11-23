<script lang="ts">
    import { onMount, tick } from "svelte";
    import { activeSessionId } from "$lib/stores/session";
    import { sendToLlama } from "$lib/services/chat";
    import { uploadDocument } from "$lib/services/documents";

    let messages: { sender: "user" | "ai"; text: string; ts?: string }[] = [];
    let userInput = "";
    let loadingHistory = false;
    let started = false;
    let prevSessionId: string | null = null;

    // New State
    let deepSearch = false;
    let isUploading = false;
    let uploadStatus = "";
    let fileInput: HTMLInputElement;

    // ✅ Load history ONLY when session ID changes
    $: if ($activeSessionId && $activeSessionId !== prevSessionId) {
        prevSessionId = $activeSessionId;
        loadSessionMessages($activeSessionId);
    }

    async function loadSessionMessages(sessionId: string) {
        loadingHistory = true;
        messages = []; // ✅ clear old session messages
        started = true;

        const res = await fetch(
            `http://127.0.0.1:8000/sessions/${sessionId}/messages`,
        );
        const data = await res.json();

        messages = data.map((m: any) => ({
            sender: m.role === "user" ? "user" : "ai",
            text: m.content,
            ts: m.ts,
        }));

        loadingHistory = false;
        await tick();
        scrollToBottom();
    }

    async function scrollToBottom() {
        await tick();
        const el = document.getElementById("chat-scroll");
        if (el) {
            el.scrollTop = el.scrollHeight;
            await tick();
            el.scrollTop = el.scrollHeight;
        }
    }

    async function sendMessage() {
        if (!userInput.trim()) return;

        const sid = $activeSessionId;
        messages = [...messages, { sender: "user", text: userInput }];

        let inputCopy = userInput;
        userInput = "";

        // Show "Thinking..." state
        const thinkingText = deepSearch
            ? "Thinking... (Deep Search Active 🧠)"
            : "Thinking...";

        messages = [...messages, { sender: "ai", text: thinkingText }];

        const reply = await sendToLlama(inputCopy, sid, deepSearch);

        // Remove the "Thinking..." message
        messages.pop();

        messages = [...messages, { sender: "ai", text: reply }];

        await tick();
        scrollToBottom();
    }

    async function handleFileUpload(e: Event) {
        const target = e.target as HTMLInputElement;
        if (!target.files || target.files.length === 0) return;

        const file = target.files[0];
        isUploading = true;
        uploadStatus = `Uploading ${file.name}...`;

        try {
            const res = await uploadDocument(file);
            uploadStatus = `✅ ${file.name} uploaded! Processing in background.`;

            // Auto-start chat if not started
            if (!started) {
                started = true;
                messages = [];
                activeSessionId.set(null);
            }

            // Add system message about upload
            messages = [
                ...messages,
                {
                    sender: "ai",
                    text: `I have received **${file.name}**. I am processing it now. You can ask me questions about it!`,
                },
            ];
            await tick();
            scrollToBottom();

            setTimeout(() => (uploadStatus = ""), 5000);
        } catch (err) {
            uploadStatus = `❌ Error uploading ${file.name}`;
        } finally {
            isUploading = false;
            target.value = ""; // reset
        }
    }
</script>

<!-- Persistent File Input -->
<input
    type="file"
    bind:this={fileInput}
    class="hidden"
    on:change={handleFileUpload}
    accept=".pdf,.docx,.txt,.md"
/>

{#if loadingHistory}
    <div class="flex h-full items-center justify-center text-gray-400">
        Loading session…
    </div>
{:else if !started}
    <div
        class="flex flex-col items-center justify-center h-full text-center space-y-4"
    >
        <h1 class="text-5xl font-bold">TanishGPT</h1>
        <p class="text-gray-600 dark:text-gray-300 max-w-md">
            Your private personal AI assistant with memory — running locally.
        </p>

        <div class="flex gap-4">
            <button
                class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                on:click={() => {
                    started = true;
                    messages = [];
                    activeSessionId.set(null); // start new chat
                }}
            >
                Start chatting
            </button>

            <button
                class="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600"
                on:click={() => fileInput.click()}
            >
                Upload Document
            </button>
        </div>
        {#if uploadStatus}
            <p class="text-sm text-gray-400">{uploadStatus}</p>
        {/if}
    </div>
{:else}
    <div class="flex flex-col h-full">
        <!-- Header / Toolbar -->
        <div
            class="p-2 border-b border-gray-800 flex justify-between items-center bg-[#0f172a] text-white"
        >
            <div class="flex items-center gap-2">
                <span class="font-bold px-2">TanishGPT</span>
                {#if uploadStatus}
                    <span class="text-xs text-green-400">{uploadStatus}</span>
                {/if}
            </div>
            <div class="flex items-center gap-3">
                <label
                    class="flex items-center gap-2 cursor-pointer text-sm select-none"
                >
                    <input
                        type="checkbox"
                        bind:checked={deepSearch}
                        class="accent-blue-500 w-4 h-4"
                    />
                    Deep Search (Graph RAG) 🧠
                </label>
                <button
                    class="text-sm bg-gray-700 px-3 py-1 rounded hover:bg-gray-600"
                    on:click={() => fileInput.click()}
                >
                    Upload
                </button>
            </div>
        </div>

        <div
            id="chat-scroll"
            class="flex-1 space-y-4 overflow-y-auto p-2 pb-28"
        >
            {#each messages as msg}
                <div class="w-full">
                    <div
                        class={`p-3 rounded-lg max-w-xl ${
                            msg.sender === "user"
                                ? "bg-blue-600 text-white ml-auto"
                                : "bg-gray-200 dark:bg-gray-800 text-black dark:text-white mr-auto"
                        }`}
                    >
                        <!-- Render markdown-like bold -->
                        {@html msg.text
                            .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
                            .replace(/\n/g, "<br>")}
                    </div>
                </div>
            {/each}
        </div>

        <div
            class="fixed bottom-0 left-0 w-full md:pl-64 bg-[#0f172a] border-t border-gray-800 p-4 flex items-center gap-3"
        >
            <input
                bind:value={userInput}
                type="text"
                placeholder={deepSearch
                    ? "Ask a deep question..."
                    : "Send a message..."}
                class="flex-1 bg-[#1e293b] text-white placeholder-gray-400 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                on:keydown={(e) => e.key === "Enter" && sendMessage()}
            />
            <button
                class="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-lg transition-all"
                on:click={sendMessage}
            >
                Send
            </button>
        </div>
    </div>
{/if}
