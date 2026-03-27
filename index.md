---
layout: default
title: Marek Oerlemans
---

## Blog posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) — {{ post.date | date: "%B %Y" }}
{% endfor %}
