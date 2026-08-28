/** Read-only build metadata row for the General settings section. */
import type { PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import css from './BuildInfoRow.module.css'

/** Full component props: the General item owner share plus the settings locale seat. */
export type BuildInfoRowProps = PropsRuntime<'settings.general.item'> & PropsLocale<'settings'>

/** Format the public build metadata injected into the client bundle. */
export function localBuildVersion(): string | undefined {
  const version = process.env.DSH_CLIENT_VERSION
  if (version === undefined) return undefined
  const commit = process.env.DSH_CLIENT_COMMIT_HASH
  return version
    + (commit === undefined ? '' : `-${commit}`)
    + (process.env.DSH_CLIENT_GIT_DIRTY === 'true' ? '-dirty' : '')
}

/**
 * Render the read-only local build metadata row.
 * @param props - composed settings-row props.
 * @returns the version row, or null when no build metadata was injected.
 */
export function BuildInfoRow({ t }: BuildInfoRowProps) {
  const version = localBuildVersion()
  if (version === undefined) return null
  return (
    <div className={css.row}>
      <div className={css.rowText}>
        <div className={css.title}>{t('buildVersion.title')}</div>
        <div className={css.desc}>{t('buildVersion.description')}</div>
      </div>
      <code className={css.value}>{version}</code>
    </div>
  )
}