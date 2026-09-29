-- Opens a centering wrapper that footer-inject.lua closes, once both the
-- social-share icons (appended between these two) and the Buy Me A Book
-- button have been appended. Must be listed BEFORE social-share in filters:
-- so this div opens before the icons are appended, and footer-inject.lua
-- must be listed AFTER social-share so it closes after both are in.
function Meta(m)
  quarto.doc.includeText("after-body", '<div style="text-align: center;">')
  return m
end
