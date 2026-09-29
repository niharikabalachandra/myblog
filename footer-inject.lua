-- Inserts footer.html via the same quarto.doc.includeText("after-body", ...)
-- mechanism social-share.lua uses, so relative order between the two is
-- controlled by filters: list order instead of YAML include-after-body
-- (which always runs before any Lua-filter-based include).
function Meta(m)
  local path = quarto.project.directory .. "/footer.html"
  local f = io.open(path, "r")
  if f then
    local footer_html = f:read("*a")
    f:close()
    quarto.doc.includeText("after-body", footer_html)
  end
  quarto.doc.includeText("after-body", "</div>")
  return m
end
