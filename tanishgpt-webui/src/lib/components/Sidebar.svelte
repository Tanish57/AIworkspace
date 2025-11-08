<script lang="ts">
    import { onMount } from "svelte";
    import { sidebarOpen } from "$lib/stores/ui";
    import { activeSessionId } from "$lib/stores/session";

    let sessions: any[] = [];

    async function loadSessions() {
        const res = await fetch("http://127.0.0.1:8000/sessions");
        sessions = await res.json();
        return sessions;
    }

    async function createNewSession() {
        const res = await fetch("http://127.0.0.1:8000/sessions/new", {
            method: "POST"
        });

        const data = await res.json();

        // ✅ Set active session ONCE
        activeSessionId.set(data.id);

        // ✅ Refresh session list
        await loadSessions();

        sidebarOpen.set(false);
    }

    onMount(loadSessions);

    function openSession(id: string) {
        activeSessionId.set(id);
        sidebarOpen.set(false);
    }
</script>

<aside
    class="fixed top-0 left-0 h-full w-64 bg-[#121826] text-white shadow-xl z-40
           transform transition-transform duration-300 md:translate-x-0"
    class:translate-x-0={$sidebarOpen}
    class:-translate-x-full={!$sidebarOpen}
>
    <div class="p-6 text-xl font-bold">TanishGPT</div>

    <button
        class="w-full text-left px-6 py-2 mb-4 bg-blue-600 hover:bg-blue-700 rounded-lg"
        on:click={createNewSession}
    >
        + New Chat
    </button>

    <nav class="space-y-1 px-4 overflow-y-auto max-h-[70vh]">
        {#each sessions as s}
        <button
            class="w-full text-left px-3 py-2 rounded hover:bg-[#1e293b] transition-all truncate text-sm
                   {s.id === $activeSessionId ? 'bg-[#1e293b]' : 'bg-transparent'}"
            on:click={() => openSession(s.id)}
        >
            {s.title}
            <div class="text-xs opacity-50">
                {new Date(s.last_active * 1000).toLocaleString()}
            </div>
        </button>
        {/each}
    </nav>

    <div class="mt-6 px-6 space-y-3">
        <a href="/memories" class="block text-sm hover:text-gray-300">Memories</a>
        <a href="/settings" class="block text-sm hover:text-gray-300">Settings</a>
    </div>
</aside>

<div
    class="fixed inset-0 bg-black/40 z-30 md:hidden"
    class:hidden={!$sidebarOpen}
    on:click={() => sidebarOpen.set(false)}
></div>
