# Keel visual identity

![Keel icon, wordmark and size variants](preview.png)

The sail carries the supplied Flow direction into a simple vector mark: a jade
leading surface, a rust underside, and an ink tile. The two surfaces meet along
one continuous curve. No texture, tiny outlines or drop shadows are needed.

| Colour | Value | Use |
| --- | --- | --- |
| Jade | `#709c8f` | Leading surface |
| Rust | `#b66e4f` | Underside |
| Ink | `#101a1c` | Icon background |
| Paper | `#f7f3e9` | Wordmark on dark surfaces |

- `icon.svg`: rounded app tile; PNG exports at 32, 128, 256, 512 and 1024px.
- `mark.svg`: transparent symbol for headers and future app integrations.
- `mark-mono.svg`: flat single-colour symbol; inherits `currentColor` inline.
- `preview.html`: inspect the mark, wordmark, colour and small-size variants.

Keep the symbol proportional. Leave at least one quarter of its width as clear
space. Use the flat mark for single-colour output. The wordmark uses the host's
sans-serif typeface and remains live text, so it needs no bundled font.

The supplied “Flow v1.2.2” board is a design reference, not a software version or
a production-release claim. These assets do not change the CLI or TUI palette.

To re-export a PNG, render `icon.svg` at its exact target size on a transparent
canvas in a browser. The supplied PNGs were rendered headlessly in Chromium.
