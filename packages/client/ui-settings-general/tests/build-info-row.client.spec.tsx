// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { BuildInfoRowProps } from '../src/client/BuildInfoRow.tsx'
import { BuildInfoRow } from '../src/client/BuildInfoRow.tsx'
import { zh } from '../src/client/locales.ts'

const unusedHook = (() => { throw new Error('unused by BuildInfoRow') }) as never
const noAttention: Parameters<Parameters<BuildInfoRowProps['useSessionPendingInteraction']>[0]>[0] = new Map()
const props: Omit<BuildInfoRowProps, 't'> = {
  useSessions: unusedHook,
  useSessionPendingInteraction: selector => selector(noAttention),
  useWorkspaces: unusedHook,
}
const t: BuildInfoRowProps['t'] = key => (zh as Record<string, string>)[key] ?? key

beforeEach(() => { vi.unstubAllEnvs() })
afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
})

describe('BuildInfoRow', () => {
  it('renders version, commit, and dirty metadata in General settings', () => {
    vi.stubEnv('DSH_CLIENT_VERSION', '1.2.3-rc.4')
    vi.stubEnv('DSH_CLIENT_COMMIT_HASH', '0123456')
    vi.stubEnv('DSH_CLIENT_GIT_DIRTY', 'true')
    render(<BuildInfoRow {...props} t={t} />)
    expect(screen.getByText('构建版本')).toBeTruthy()
    expect(screen.getByText('当前本地构建的版本与提交状态')).toBeTruthy()
    expect(screen.getByText('1.2.3-rc.4-0123456-dirty')).toBeTruthy()
  })

  it('renders nothing when the build did not provide a version', () => {
    vi.stubEnv('DSH_CLIENT_VERSION', undefined as unknown as string)
    const { container } = render(<BuildInfoRow {...props} t={t} />)
    expect(container.firstChild).toBeNull()
  })
})