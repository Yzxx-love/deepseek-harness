# Agent Note: DeepSeek visual field from the supplied UI reference

Status: implemented


English | [中文](2026-08-29-deepseek-background-visual-field.zh.md)
## Problem

The supplied dsh-client-ui-deepseek-bg package uses a dark-only aurora, constellation grid, particle whale, and frosted surfaces, but its selectors target one generated DSH build and are not stable source-level integration points. The local Harness also needs to keep its official logo, slots, routes, and light theme behavior.

## Decision

Move the visual field into the existing ui-theme client lifecycle. Add a dependency-free Canvas renderer for the aurora, 90px constellation grid, and lightweight particle whale; add dark-only frosted surface rules; and mark the frame, sidebar, details, conversation, assistant, and user-bubble surfaces with stable data-ui-surface attributes. Keep the light palette unchanged. Pause rendering while hidden, throttle to 30fps, and honor reduced-motion and reduced-transparency preferences.

## Alternatives considered

**Copy the reference plugin unchanged.** Rejected because its runtime class selectors are build-specific and could miss the current DSH source components.

**Add a separate optional skin bundle.** Rejected because this checkout's default Web composition already owns ui-theme; keeping the field there means the running local build receives it without a second profile edit.

**Apply the dark material to both themes.** Rejected because the supplied reference explicitly keeps the official light theme unchanged and users need a reliable light-mode fallback.

## Consequences

The default Web UI now loads the reference-inspired field from the normal ui-theme bundle. The background remains decorative and pointer-transparent, so navigation, resizing, composer input, settings, and message actions keep their existing interaction paths. The official DSH logo and build metadata placement stay unchanged. GUI tests remain green after the intentional sidebar snapshot update; real Web replay still has pre-existing scroll/settings fixture failures that are reported separately.
