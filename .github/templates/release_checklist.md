---
title: "Release Checklist: {{ payload.release.tag_name }}"
---

## Release {{ payload.release.tag_name }} Checklist

- [ ] Changelog updated
- [ ] All tests passing
- [ ] Version number bumped in `__init__.py` and `pyproject.toml`
- [ ] Wheels build successful
- [ ] Release announcement drafted

## Post-release

- Push wheels to PyPI and NVIDIA PyPI