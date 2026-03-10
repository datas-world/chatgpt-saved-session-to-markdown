# Social Preview

| Field | Value |
|-------|-------|
| **File** | `.github/social-preview.png` |
| **Dimensions** | 1280×640 px |
| **Design system** | Operational Taxonomy (datas-world) |
| **Generated** | 2026-03-10 via `canvas-design` AI skill |
| **Background** | `#0f172a` (Tailwind `slate-900`) |
| **Accent colour** | cyan |

## About this image

This repository provides a conversion tool that transforms exported ChatGPT conversation JSON files into clean, well-structured Markdown documents, preserving message threading, code blocks, and metadata in a portable format. The social preview uses **conversation thread geometry** — alternating left/right message-bubble polygons in a vertical thread — to directly represent the tool's input: a saved chat conversation rendered as geometry. The cyan accent reflects the tool's ChatGPT origin and the freshness of extracted knowledge.

## Design system

The **Operational Taxonomy** design language is a unified visual identity for all `datas-world` repositories. Every social preview shares a dark navy background (`#0f172a`, Tailwind CSS `slate-900`) on a 1280×640 px canvas — GitHub's recommended social preview dimensions — with a 40 px safe zone on all sides to handle cropping in circular or square contexts.

Typography is intentionally minimal: thin/light-weight geometric sans-serif lettering acts as a visual accent rather than a primary information carrier. The real communication happens through **geometry and colour**: radial patterns, topological layouts, and layered shapes that map each repository's domain into a visual language. Information lives in structure, not prose — hence the name *Operational Taxonomy*.

Accent colours are drawn from the org's [canonical 22-label colour palette](https://github.com/datas-world/.github/blob/main/labels.json), ensuring visual consistency with the issue-tracker taxonomy used across every repository. This creates a coherent brand thread from repository labels through social cards to documentation.

## Updating this image

1. **Regenerate the PNG** using the [`canvas-design` AI skill](https://github.com/anthropics/anthropic-tools) in a GitHub Copilot CLI session:

   ```
   @copilot /use-skill canvas-design
   ```

   Provide the prompt: *"1280×640 px social preview for `datas-world/chatgpt-saved-session-to-markdown` — dark navy `#0f172a` background, cyan accent, [describe your desired motif changes], 40 px safe zone, Operational Taxonomy style."*

2. **Download the generated PNG** and rename it `social-preview.png`.

3. **Upload via GitHub Settings UI:**
   - Navigate to `https://github.com/datas-world/chatgpt-saved-session-to-markdown/settings`
   - Scroll to **Social preview** under the *Features* section
   - Click **Edit** → **Upload an image** → select `social-preview.png`
   - Click **Save changes**

4. **Commit the PNG to `.github/`** so the file stays in version control:

   ```bash
   git add .github/social-preview.png
   git commit -m "design(social-preview): update social preview image"
   git push
   ```

5. **Update this file** if the motif, accent colour, or generation method changes.

## References

| Resource | URL |
|----------|-----|
| GitHub social preview docs | [Customising your repository's social media preview](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview) |
| How to upload the image in GitHub UI | [Adding a social media preview image](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview#adding-a-social-media-preview-image-to-your-repository) |
| GitHub Markdown syntax reference | [Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) |
| Tailwind CSS slate palette | [Customising colours — slate-900 = `#0f172a`](https://tailwindcss.com/docs/customizing-colors#default-color-palette) |
| datas-world canonical label colours | [`labels.json`](https://github.com/datas-world/.github/blob/main/labels.json) |
| Image dimensions spec | [GitHub Open Graph image: 1280×640 px (min 640×320 px)](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview) |
