<script lang="ts">
    import "../app.css";
    import { sidebarOpen } from "$lib/stores/ui";
    import Sidebar from "$lib/components/Sidebar.svelte";
</script>

<!-- SIDEBAR (always mounted) -->
<Sidebar />

<!-- MOBILE HAMBURGER -->
<button
    on:click={() => sidebarOpen.update(v => !v)}
    class="md:hidden fixed top-4 left-4 z-50 p-3 rounded-lg bg-[#1E293B] hover:bg-[#334155] text-white"
>
    <!-- Icon -->
    <svg
        xmlns="http://www.w3.org/2000/svg"
        class="h-6 w-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
    >
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M4 6h16M4 12h16M4 18h16" />
    </svg>
</button>

<!-- MAIN CONTENT (padding-left on desktop) -->
<div class="md:pl-64 pt-10 px-6">
    <slot />
</div>
