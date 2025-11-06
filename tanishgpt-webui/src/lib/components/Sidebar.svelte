<script lang="ts">
    import { sidebarOpen } from "$lib/stores/ui";
</script>

<!-- SIDEBAR CONTAINER -->
<div
    class="
        fixed top-0 left-0 h-full w-64 bg-[#121826] text-white shadow-xl z-40
        transform transition-transform duration-300
        md:translate-x-0
    "
    class:translate-x-0={$sidebarOpen}
    class:-translate-x-full={!$sidebarOpen}
>
    <div class="p-6 text-xl font-bold">TanishGPT</div>

    <nav class="space-y-4 px-6">
        <a href="/" class="block">New Chat</a>
        <a href="/memories" class="block">Memories</a>
        <a href="/settings" class="block">Settings</a>
    </nav>
</div>

<!-- BACKDROP (mobile only) -->
<div
    class="fixed inset-0 bg-black/40 z-30 md:hidden"
    class:hidden={!$sidebarOpen}
    on:click={() => sidebarOpen.set(false)}
></div>
