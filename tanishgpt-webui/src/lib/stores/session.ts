import { writable } from "svelte/store";

export const activeSessionId = writable<string | null>(null);

// This will help us keep track of which session is currently opened