# Agent Note: Keep the official brand stable and move local build metadata to Settings

Status: implemented

English | [中文](2026-08-29-sidebar-brand-build-version-settings.zh.md)
## Problem

The local-build fallback stacked DSH Local Build and a version-commit-dirty badge beside the sidebar brand. That metadata made the navigation identity change between build modes and consumed the narrowest part of the application chrome.

## Decision

The sidebar fallback renders the official fish mark and DeepSeek Harness wordmark through the existing BrandWordmark primitive. It no longer reads or formats DSH_CLIENT_VERSION, DSH_CLIENT_COMMIT_HASH, or DSH_CLIENT_GIT_DIRTY. ui-settings-general registers a read-only Build version row in Settings → General and formats the same public metadata there; the row is absent when no version metadata was injected.

## Alternatives considered

**Keep the badge beside the brand.** Rejected because build provenance is diagnostic metadata, not navigation identity, and the two-line badge crowds the sidebar.

**Remove the metadata entirely.** Rejected because local builds still need a visible version and commit state for support and reproduction; Settings provides a stable, discoverable home.

**Switch every local build to the official build profile.** Rejected because the profile controls slot registration for deployment identity; changing the fallback keeps local and official profiles composable while giving both the same visible brand.

## Consequences

The sidebar keeps a stable official identity in local and official builds. Settings exposes the public build version, optional short commit, and dirty marker without adding a new persistence field or exposing credentials. Tests pin both the absence of the sidebar badge and the General row formatting; the row remains presentation-only and never reaches model requests.
