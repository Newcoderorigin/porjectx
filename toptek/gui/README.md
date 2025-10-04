# Toptek GUI Theme

The desktop mission control now ships with a dark dashboard theme that is driven by
semantic colour tokens. The palette is shared by `toptek/gui/app.py` (style
registration) and `toptek/gui/widgets.py` (widget construction) through the
constants exposed in `toptek/gui/__init__.py`.

## Palette tokens

| Token | Hex | Usage |
| --- | --- | --- |
| `canvas` | `#0b1120` | Application background, notebook, and root windows. |
| `surface` | `#111827` | Raised panels such as section frames and dashboard cards. |
| `surface_alt` | `#1e293b` | Text fields, multi-line editors, and hover states. |
| `surface_muted` | `#18243a` | Notebook tab hover colour and pressed neutral buttons. |
| `border` | `#1f2937` | Card borders and section outlines. |
| `border_muted` | `#243047` | Chart container outlines and disabled controls. |
| `accent` | `#8b5cf6` | Primary call-to-action buttons and metric values. |
| `accent_hover` | `#a855f7` | Hover colour for accent buttons. |
| `accent_active` | `#7c3aed` | Pressed colour for accent buttons. |
| `accent_alt` | `#38bdf8` | Informational highlights (status text, progress fills). |
| `text` | `#e2e8f0` | Primary body text. |
| `text_muted` | `#94a3b8` | Subdued captions and helper copy. |
| `success` | `#22c55e` | Positive status indicators (verification, guard OK). |
| `warning` | `#f97316` | Heads-up alerts that require follow-up action. |
| `danger` | `#f87171` | Blocking errors or defensive guard states. |

## Tk styles

`launch_app` registers the reusable ttk styles that every tab consumes:

- **Layout styles** – `DashboardBackground.TFrame`, `AppContainer.TFrame`,
  `Section.TLabelframe`, `DashboardCard.TFrame`, and `ChartContainer.TFrame`
  align structural backgrounds and borders across the notebook.
- **Typography styles** – `Header.TLabel`, `SubHeader.TLabel`, `Body.TLabel`,
  `Surface.TLabel`, `CardHeading.TLabel`, `MetricValue.TLabel`, and
  `MetricCaption.TLabel` keep copy aligned with the light-on-dark palette.
- **Status styles** – `StatusInfo.TLabel` (for canvas backgrounds) and
  `SurfaceStatus.TLabel` (for raised surfaces) hold highlights such as
  verification outcomes and guard readiness.
- **Input styles** – `Input.TEntry`, `Input.TCombobox`, `Input.TSpinbox`,
  `Input.TRadiobutton`, and `Input.TCheckbutton` provide consistent field
  surfaces while default buttons consume either `Accent.TButton` (primary
  actions) or `Neutral.TButton` (secondary actions).
- **Feedback styles** – `Accent.Horizontal.TProgressbar` colours the mission
  checklist progress bar and the notebook tabs receive hover/active maps for
  modern feedback.

`BaseTab.style_text_widget` applies the `TEXT_WIDGET_DEFAULTS` tokens to every
`tk.Text` instance so scrollable panes match the rest of the experience.

## Usage guidelines

1. **Choose the right surface** – Wrap new tab sections in
   `Section.TLabelframe` when you need a contained card; use
   `DashboardBackground.TFrame` for neutral layouts.
2. **Buttons** – Reserve `Accent.TButton` for primary calls-to-action and
   fall back to `Neutral.TButton` for supportive controls. Both have hover and
   pressed colour maps baked in.
3. **Text entries** – Always request the `Input.*` styles on ttk entry,
   combobox, spinbox, radio, or checkbutton widgets so field backgrounds stay
   coordinated with the palette.
4. **Statuses** – Prefer the status styles rather than setting literal colours.
   For context-specific overrides (e.g., success/danger), reuse the
   `DARK_PALETTE` tokens from `toptek/gui/__init__.py`.
5. **Dashboard extensions** – New cards in `DashboardTab` should follow the
   existing pattern: `DashboardCard.TFrame` containers with a heading label,
   `MetricValue.TLabel` for the primary figure, and `MetricCaption.TLabel` for
   the descriptive copy. Charts and rich panes should sit inside
   `ChartContainer.TFrame` to inherit border and padding rules.

## Extending the theme

- Extend `DARK_PALETTE` when introducing additional semantic colours and update
  both this table and any affected ttk styles.
- Keep ttk style registration centralised in `launch_app` so the theme remains
  declarative and easy to audit.
- When adding new text widgets, call `BaseTab.style_text_widget` immediately
  after instantiation to apply the shared configuration.
- Run the linter/test suite (`ruff`, `black`, `mypy`, `pytest`) after theme
  changes to guard against regressions—the project CI expects these gates.

