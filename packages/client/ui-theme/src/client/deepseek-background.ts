/**
 * DeepSeek Harness visual field.
 *
 * A dependency-free adaptation of the supplied dsh-client-ui-deepseek-bg
 * plugin. It keeps the reference's dark-only aurora, constellation grid and
 * particle whale language, but owns the lifecycle from ui-theme.
 */
import type { Context } from '@deepseek-ai/cordis'

const ROOT_ID = 'dsh-deepseek-bg'
const PLUGIN_ID = '@deepseek-ai/dsh-client-ui-theme'
const FRAME_MS = 1000 / 30
const GRID_SIZE = 90
const INTERACTION_RADIUS = 140

interface Dot { restX: number; restY: number; x: number; y: number; vx: number; vy: number }
interface Particle { x: number; y: number; size: number; phase: number }

function isDarkTheme(): boolean {
  const body = document.body
  const root = document.documentElement
  return body?.hasAttribute('data-ds-dark-theme') === true
    || root.hasAttribute('data-ds-dark-theme')
    || root.dataset.theme === 'dark'
}

function cssColor(name: string, fallback: string): string {
  const value = getComputedStyle(document.body).getPropertyValue(name).trim()
  return value === '' ? fallback : value
}

function createElement<K extends keyof HTMLElementTagNameMap>(tag: K, className: string): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag)
  element.className = className
  return element
}

function seed(value: number): number {
  const x = Math.sin(value * 12.9898) * 43758.5453
  return x - Math.floor(x)
}

function buildWhaleParticles(): Particle[] {
  const particles: Particle[] = []
  let index = 1
  while (particles.length < 240 && index < 4_000) {
    const x = seed(index) * 2 - 1
    const y = seed(index + 17) * 2 - 1
    const body = x * x + y * y * 2.6 < 0.82
    const tail = x > 0.64 && Math.abs(y) < (1 - x) * 1.45
    const fin = x < -0.25 && y > 0.45 && y < 0.85 + x * 0.25
    if (body || tail || fin) {
      particles.push({ x, y, size: 0.7 + seed(index + 31) * 1.5, phase: seed(index + 47) * Math.PI * 2 })
    }
    index += 1
  }
  return particles
}

function drawWhale(context: CanvasRenderingContext2D, width: number, height: number, particles: readonly Particle[], time: number, mouse: { x: number; y: number; active: boolean }): void {
  context.clearRect(0, 0, width, height)
  const scale = Math.min(width * 0.32, height * 0.48, 330)
  const centerX = width * 0.57
  const centerY = height * 0.38
  const lightX = mouse.active ? mouse.x : width * 0.72
  const lightY = mouse.active ? mouse.y : height * 0.28
  context.fillStyle = cssColor('--dsh-deepseek-whale', 'rgb(176 205 255)')
  context.shadowColor = cssColor('--dsh-deepseek-whale-glow', 'rgb(87 143 255 / .65)')
  context.shadowBlur = 18
  for (const particle of particles) {
    const drift = Math.sin(time * 0.0012 + particle.phase) * 2
    const x = centerX + particle.x * scale + drift
    const y = centerY + particle.y * scale * 0.62
    const light = Math.max(0.18, 1 - Math.hypot(x - lightX, y - lightY) / (scale * 0.9))
    context.globalAlpha = 0.18 + light * 0.72
    context.fillRect(x, y, particle.size * (0.75 + light * 0.75), particle.size * (0.75 + light * 0.75))
  }
  context.globalAlpha = 1
  context.shadowBlur = 0
}

function drawAurora(context: CanvasRenderingContext2D, width: number, height: number, time: number): void {
  context.clearRect(0, 0, width, height)
  const phase = time * 0.00012
  const blobs = [
    { x: 0.2 + Math.sin(phase) * 0.08, y: 0.22, radius: 0.7, color: '--dsh-deepseek-aurora-blue' },
    { x: 0.75, y: 0.2 + Math.cos(phase * 1.4) * 0.1, radius: 0.52, color: '--dsh-deepseek-aurora-gold' },
    { x: 0.54 + Math.sin(phase * 0.8) * 0.1, y: 0.64, radius: 0.65, color: '--dsh-deepseek-aurora-indigo' },
  ]
  for (const blob of blobs) {
    const x = width * blob.x
    const y = height * blob.y
    const gradient = context.createRadialGradient(x, y, 0, x, y, Math.max(width, height) * blob.radius)
    const color = cssColor(blob.color, 'rgb(61 117 207)')
    gradient.addColorStop(0, color.replace(')', ' / .38)'))
    gradient.addColorStop(0.42, color.replace(')', ' / .12)'))
    gradient.addColorStop(1, 'transparent')
    context.fillStyle = gradient
    context.fillRect(0, 0, width, height)
  }
}

function drawConstellation(context: CanvasRenderingContext2D, width: number, height: number, dots: readonly Dot[], columns: number, rows: number, mouse: { x: number; y: number; active: boolean }): void {
  context.clearRect(0, 0, width, height)
  context.lineWidth = 0.5
  context.strokeStyle = cssColor('--dsh-deepseek-constellation-line', 'rgb(255 255 255 / .11)')
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const left = dots[row * columns + column]
      const right = dots[row * columns + column + 1]
      if (left === undefined || right === undefined) continue
      context.beginPath(); context.moveTo(left.x, left.y); context.lineTo(right.x, right.y); context.stroke()
    }
  }
  for (let column = 0; column < columns; column += 1) {
    for (let row = 0; row < rows - 1; row += 1) {
      const top = dots[row * columns + column]
      const bottom = dots[(row + 1) * columns + column]
      if (top === undefined || bottom === undefined) continue
      context.beginPath(); context.moveTo(top.x, top.y); context.lineTo(bottom.x, bottom.y); context.stroke()
    }
  }
  context.fillStyle = cssColor('--dsh-deepseek-constellation-dot', 'rgb(255 255 255 / .19)')
  for (const dot of dots) {
    const distance = mouse.active ? Math.hypot(dot.x - mouse.x, dot.y - mouse.y) : Infinity
    const emphasis = distance < INTERACTION_RADIUS ? 1 - distance / INTERACTION_RADIUS : 0
    context.globalAlpha = 0.42 + emphasis * 0.58
    context.beginPath(); context.arc(dot.x, dot.y, 1.5 + emphasis * 2, 0, Math.PI * 2); context.fill()
  }
  context.globalAlpha = 1
}

/** Mount the dark-only visual field for the ui-theme plugin. */
export function installDeepSeekBackground(ctx: Context): void {
  if (typeof document === 'undefined') return
  ctx.effect(() => {
    const host = document.body
    if (host === null) return () => {}
    const existing = document.getElementById(ROOT_ID)
    const root = existing ?? createElement('div', 'dsh-deepseek-bg')
    const owned = existing === null
    root.id = ROOT_ID
    root.dataset.plugin = PLUGIN_ID
    root.setAttribute('aria-hidden', 'true')
    if (owned) {
      root.append(
        createElement('div', 'dsh-deepseek-bg-orb dsh-deepseek-bg-orb-a'),
        createElement('div', 'dsh-deepseek-bg-orb dsh-deepseek-bg-orb-b'),
        createElement('div', 'dsh-deepseek-bg-orb dsh-deepseek-bg-orb-c'),
      )
      root.append(
        createElement('canvas', 'dsh-deepseek-bg-aurora'),
        createElement('canvas', 'dsh-deepseek-bg-constellation'),
        createElement('canvas', 'dsh-deepseek-bg-whale'),
      )
      host.appendChild(root)
    }
    // jsdom intentionally omits a canvas implementation; leave the DOM clean in unit-test lanes.
    if (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent)) return () => { if (owned) root.remove() }
    const aurora = root.querySelector<HTMLCanvasElement>('.dsh-deepseek-bg-aurora')
    const constellation = root.querySelector<HTMLCanvasElement>('.dsh-deepseek-bg-constellation')
    const whale = root.querySelector<HTMLCanvasElement>('.dsh-deepseek-bg-whale')
    const auroraContext = aurora?.getContext('2d') ?? null
    const constellationContext = constellation?.getContext('2d') ?? null
    const whaleContext = whale?.getContext('2d') ?? null
    if (aurora === null || constellation === null || whale === null || auroraContext === null || constellationContext === null || whaleContext === null) {
      return () => { if (owned) root.remove() }
    }
    const reducedMotion = typeof matchMedia === 'undefined' ? false : matchMedia('(prefers-reduced-motion: reduce)').matches
    const whaleParticles = buildWhaleParticles()
    const dots: Dot[] = []
    const mouse = { x: 0, y: 0, active: false }
    let columns = 0; let rows = 0; let frame = 0; let lastFrame = 0; let width = 0; let height = 0
    const resize = (): void => {
      width = window.innerWidth; height = window.innerHeight
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
      for (const canvas of [aurora, constellation, whale]) {
        canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr)
        canvas.style.width = width + 'px'; canvas.style.height = height + 'px'
      }
      auroraContext.setTransform(dpr, 0, 0, dpr, 0, 0); constellationContext.setTransform(dpr, 0, 0, dpr, 0, 0); whaleContext.setTransform(dpr, 0, 0, dpr, 0, 0)
      columns = Math.ceil(width / GRID_SIZE) + 1; rows = Math.ceil(height / GRID_SIZE) + 1; dots.length = 0
      const offsetX = (width - (columns - 1) * GRID_SIZE) / 2; const offsetY = (height - (rows - 1) * GRID_SIZE) / 2
      for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
        const x = offsetX + column * GRID_SIZE; const y = offsetY + row * GRID_SIZE
        dots.push({ restX: x, restY: y, x, y, vx: 0, vy: 0 })
      }
    }
    const applyTheme = (): void => {
      const dark = isDarkTheme()
      root.classList.toggle('dsh-deepseek-bg-dark', dark)
      if (dark && frame === 0 && !reducedMotion) frame = requestAnimationFrame(draw)
      if (!dark && frame !== 0) { cancelAnimationFrame(frame); frame = 0 }
    }
    const draw = (now: number): void => {
      if (!isDarkTheme()) {
        auroraContext.clearRect(0, 0, width, height); constellationContext.clearRect(0, 0, width, height); whaleContext.clearRect(0, 0, width, height)
        frame = 0; return
      }
      if (now - lastFrame < FRAME_MS) { frame = requestAnimationFrame(draw); return }
      lastFrame = now; drawAurora(auroraContext, width, height, now)
      for (const dot of dots) {
        if (mouse.active) {
          const dx = dot.x - mouse.x; const dy = dot.y - mouse.y; const distance = Math.hypot(dx, dy)
          if (distance > 0.1 && distance < INTERACTION_RADIUS) {
            const force = (1 - distance / INTERACTION_RADIUS) * 1.8; dot.vx += dx / distance * force; dot.vy += dy / distance * force
          }
        }
        dot.vx += (dot.restX - dot.x) * 0.05; dot.vy += (dot.restY - dot.y) * 0.05; dot.vx *= 0.84; dot.vy *= 0.84; dot.x += dot.vx; dot.y += dot.vy
      }
      drawConstellation(constellationContext, width, height, dots, columns, rows, mouse); drawWhale(whaleContext, width, height, whaleParticles, now, mouse)
      frame = reducedMotion ? 0 : requestAnimationFrame(draw)
    }
    const onPointerMove = (event: PointerEvent): void => { mouse.x = event.clientX; mouse.y = event.clientY; mouse.active = true }
    const onPointerLeave = (): void => { mouse.active = false }
    const onVisibility = (): void => {
      if (document.visibilityState === 'hidden' && frame !== 0) { cancelAnimationFrame(frame); frame = 0 }
      else if (document.visibilityState === 'visible' && frame === 0 && !reducedMotion) frame = requestAnimationFrame(draw)
    }
    const media = typeof matchMedia === 'undefined' ? undefined : matchMedia('(prefers-reduced-motion: reduce)')
    const onMediaChange = (): void => { if (frame !== 0) cancelAnimationFrame(frame); frame = 0; draw(performance.now()) }
    const observer = new MutationObserver(applyTheme)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-ds-dark-theme', 'data-ds-light-theme', 'data-theme', 'class'] })
    if (document.body !== null) observer.observe(document.body, { attributes: true, attributeFilter: ['data-ds-dark-theme', 'data-ds-light-theme'] })
    window.addEventListener('resize', resize, { passive: true }); window.addEventListener('pointermove', onPointerMove, { passive: true }); window.addEventListener('mouseleave', onPointerLeave); document.addEventListener('visibilitychange', onVisibility); media?.addEventListener('change', onMediaChange)
    resize(); applyTheme(); draw(performance.now())
    return () => {
      if (frame !== 0) cancelAnimationFrame(frame)
      observer.disconnect(); window.removeEventListener('resize', resize); window.removeEventListener('pointermove', onPointerMove); window.removeEventListener('mouseleave', onPointerLeave); document.removeEventListener('visibilitychange', onVisibility); media?.removeEventListener('change', onMediaChange)
      if (owned) root.remove()
    }
  }, 'ui-theme: DeepSeek visual field')
}
