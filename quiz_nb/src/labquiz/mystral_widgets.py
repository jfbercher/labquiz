"""
mystral_widgets.py — Minimal ipywidgets-compatible shim for Mystral Editor.
Backed by real DOM elements via Pyodide's js/ffi bridge.

Usage in labquiz:
    from . import mystral_widgets as widgets
    from .mystral_widgets import display, Markdown, Javascript
"""
import sys
import re

import js
from pyodide.ffi import create_proxy

# ---------------------------------------------------------------------------
# Output container helpers
# ---------------------------------------------------------------------------

def _get_output_container():
    """Return the DOM node designated for widget output in the current cell."""
    try:
        out = js.globalThis.__mystral_cell_output
        if out and out.nodeType:
            return out
    except Exception:
        pass
    return js.document.body


# Stack for nested  `with output:` context managers
_output_stack = []


def _current_output():
    """Return the innermost active Output context, or None."""
    return _output_stack[-1] if _output_stack else None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class Layout:
    """Maps Python-style CSS keyword args to DOM element style properties."""

    def __init__(self, **kwargs):
        object.__setattr__(self, '_dom', None)
        object.__setattr__(self, '_props', {})
        for k, v in kwargs.items():
            object.__getattribute__(self, '_props')[k.replace('_', '-')] = v

    def _attach(self, dom):
        object.__setattr__(self, '_dom', dom)
        props = object.__getattribute__(self, '_props')
        for k, v in props.items():
            if v is not None:
                dom.style.setProperty(k, str(v))

    def __setattr__(self, name, value):
        css_key = name.replace('_', '-')
        props = object.__getattribute__(self, '_props')
        props[css_key] = value
        dom = object.__getattribute__(self, '_dom')
        if dom is not None:
            if value is not None:
                dom.style.setProperty(css_key, str(value))
            else:
                dom.style.removeProperty(css_key)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        props = object.__getattribute__(self, '_props')
        return props.get(name.replace('_', '-'))


# ---------------------------------------------------------------------------
# Base widget
# ---------------------------------------------------------------------------

class _Widget:
    def __init__(self):
        self._dom = None
        self._layout = Layout()
        self._observers = {}
        self._proxies = []   # keep create_proxy refs alive to prevent GC

    @property
    def layout(self):
        if self._dom is not None and self._layout._dom is None:
            self._layout._attach(self._dom)
        return self._layout

    @layout.setter
    def layout(self, new_layout):
        object.__setattr__(self, '_layout', new_layout)
        if self._dom is not None:
            new_layout._attach(self._dom)

    def add_class(self, cls):
        if self._dom is not None:
            self._dom.classList.add(cls)

    def remove_class(self, cls):
        if self._dom is not None:
            self._dom.classList.remove(cls)

    def observe(self, fn, names=None):
        if isinstance(names, str):
            names = [names]
        for name in (names or ['value']):
            self._observers.setdefault(name, []).append(fn)

    def _notify(self, name, old, new):
        change = {'name': name, 'old': old, 'new': new, 'owner': self}
        for fn in self._observers.get(name, []):
            try:
                fn(change)
            except Exception as exc:
                print(f"[mystral_widgets] observer error: {exc}")

    def _build_dom(self):
        raise NotImplementedError

    def _render(self):
        if self._dom is None:
            self._build_dom()
            if self._layout is not None and self._layout._dom is None:
                self._layout._attach(self._dom)
        return self._dom


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

class Text(_Widget):
    def __init__(self, value='', placeholder='', description='',
                 style=None, layout=None, continuous_update=True, **kwargs):
        super().__init__()
        self._value = value
        self._placeholder = placeholder
        self._description = description
        self._continuous_update = continuous_update
        self._on_submit_callbacks = []
        if layout is not None:
            self._layout = layout

    @property
    def value(self):
        if self._dom is not None:
            inp = self._dom.querySelector('input')
            if inp:
                self._value = inp.value
        return self._value

    @value.setter
    def value(self, v):
        old = self._value
        self._value = v
        if self._dom is not None:
            inp = self._dom.querySelector('input')
            if inp:
                inp.value = v
        if old != v:
            self._notify('value', old, v)

    def on_submit(self, fn):
        self._on_submit_callbacks.append(fn)

    def _build_dom(self):
        wrapper = js.document.createElement('div')
        wrapper.className = 'mw-text'
        wrapper.style.setProperty('display', 'inline-flex')
        wrapper.style.setProperty('align-items', 'center')
        wrapper.style.setProperty('gap', '0.35rem')
        if self._description:
            lbl = js.document.createElement('label')
            lbl.textContent = self._description
            lbl.className = 'mw-label'
            lbl.style.setProperty('font-size', '0.82rem')
            lbl.style.setProperty('white-space', 'nowrap')
            wrapper.appendChild(lbl)
        inp = js.document.createElement('input')
        inp.type = 'text'
        inp.value = self._value
        inp.placeholder = self._placeholder
        inp.className = 'mw-input'
        inp.style.setProperty('padding', '0.3rem 0.5rem')
        inp.style.setProperty('border', '1px solid #d0d7de')
        inp.style.setProperty('border-radius', '4px')
        inp.style.setProperty('font', 'inherit')
        inp.style.setProperty('font-size', '0.85rem')

        def _on_input(evt):
            old = self._value
            self._value = inp.value
            if old != self._value:
                self._notify('value', old, self._value)

        def _on_keydown(evt):
            if evt.key == 'Enter':
                for fn in self._on_submit_callbacks:
                    try:
                        fn(self)
                    except Exception as e:
                        print(f"[mystral_widgets] on_submit error: {e}")

        p1 = create_proxy(_on_input)
        p2 = create_proxy(_on_keydown)
        self._proxies.extend([p1, p2])
        inp.addEventListener('input', p1)
        inp.addEventListener('keydown', p2)
        wrapper.appendChild(inp)
        self._dom = wrapper


# ---------------------------------------------------------------------------
# Dropdown
# ---------------------------------------------------------------------------

class Dropdown(_Widget):
    def __init__(self, options=None, value=None, description='',
                 style=None, layout=None, **kwargs):
        super().__init__()
        self._options = list(options or [])
        self._value = value if value is not None else (self._options[0] if self._options else None)
        self._description = description
        if layout is not None:
            self._layout = layout

    @property
    def value(self):
        if self._dom is not None:
            sel = self._dom.querySelector('select')
            if sel:
                self._value = sel.value
        return self._value

    @value.setter
    def value(self, v):
        old = self._value
        self._value = v
        if self._dom is not None:
            sel = self._dom.querySelector('select')
            if sel:
                sel.value = str(v) if v is not None else ''
        if old != v:
            self._notify('value', old, v)

    def _build_dom(self):
        wrapper = js.document.createElement('div')
        wrapper.className = 'mw-dropdown'
        wrapper.style.setProperty('display', 'inline-flex')
        wrapper.style.setProperty('align-items', 'center')
        wrapper.style.setProperty('gap', '0.35rem')
        if self._description:
            lbl = js.document.createElement('label')
            lbl.textContent = self._description
            lbl.className = 'mw-label'
            wrapper.appendChild(lbl)
        sel = js.document.createElement('select')
        sel.className = 'mw-select'
        sel.style.setProperty('padding', '0.3rem 0.5rem')
        sel.style.setProperty('border', '1px solid #d0d7de')
        sel.style.setProperty('border-radius', '4px')
        sel.style.setProperty('font', 'inherit')
        sel.style.setProperty('font-size', '0.85rem')
        for opt in self._options:
            o = js.document.createElement('option')
            o.value = str(opt)
            o.textContent = str(opt)
            sel.appendChild(o)
        if self._value is not None:
            sel.value = str(self._value)

        def _on_change(evt):
            old = self._value
            self._value = sel.value
            if old != self._value:
                self._notify('value', old, self._value)

        p = create_proxy(_on_change)
        self._proxies.append(p)
        sel.addEventListener('change', p)
        wrapper.appendChild(sel)
        self._dom = wrapper


# ---------------------------------------------------------------------------
# Checkbox
# ---------------------------------------------------------------------------

class Checkbox(_Widget):
    def __init__(self, value=False, description='', indent=True,
                 layout=None, **kwargs):
        super().__init__()
        self._value = value
        self._description = description
        if layout is not None:
            self._layout = layout

    @property
    def value(self):
        if self._dom is not None:
            cb = self._dom.querySelector('input[type=checkbox]')
            if cb:
                self._value = bool(cb.checked)
        return self._value

    @value.setter
    def value(self, v):
        old = self._value
        self._value = bool(v)
        if self._dom is not None:
            cb = self._dom.querySelector('input[type=checkbox]')
            if cb:
                cb.checked = self._value
        if old != self._value:
            self._notify('value', old, self._value)

    def _build_dom(self):
        wrapper = js.document.createElement('div')
        wrapper.className = 'mw-checkbox'
        wrapper.style.setProperty('display', 'inline-flex')
        wrapper.style.setProperty('align-items', 'center')
        wrapper.style.setProperty('gap', '0.3rem')
        wrapper.style.setProperty('font-size', '0.85rem')
        cb = js.document.createElement('input')
        cb.type = 'checkbox'
        cb.checked = self._value

        def _on_change(evt):
            old = self._value
            self._value = bool(cb.checked)
            if old != self._value:
                self._notify('value', old, self._value)

        p = create_proxy(_on_change)
        self._proxies.append(p)
        cb.addEventListener('change', p)
        wrapper.appendChild(cb)
        if self._description:
            lbl = js.document.createElement('label')
            lbl.textContent = self._description
            wrapper.appendChild(lbl)
        self._dom = wrapper


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------

_BUTTON_STYLE_MAP = {
    'primary':  'mw-btn-primary',
    'success':  'mw-btn-success',
    'info':     'mw-btn-info',
    'warning':  'mw-btn-warning',
    'danger':   'mw-btn-danger',
    '':         'mw-btn-default',
}


class Button(_Widget):
    def __init__(self, description='', button_style='', icon='',
                 layout=None, disabled=False, **kwargs):
        super().__init__()
        self._description = description
        self._button_style = button_style
        self._icon = icon
        self._disabled = disabled
        self._click_callbacks = []
        if layout is not None:
            self._layout = layout

    @property
    def disabled(self):
        if self._dom is not None:
            self._disabled = bool(self._dom.disabled)
        return self._disabled

    @disabled.setter
    def disabled(self, v):
        self._disabled = bool(v)
        if self._dom is not None:
            self._dom.disabled = self._disabled

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, v):
        self._description = v
        if self._dom is not None:
            self._dom.textContent = v

    def on_click(self, fn):
        self._click_callbacks.append(fn)

    def _build_dom(self):
        btn = js.document.createElement('button')
        btn.className = 'mw-btn ' + _BUTTON_STYLE_MAP.get(self._button_style, 'mw-btn-default')
        btn.textContent = self._description
        btn.disabled = self._disabled
        # base styles
        btn.style.setProperty('display', 'inline-flex')
        btn.style.setProperty('align-items', 'center')
        btn.style.setProperty('gap', '0.3rem')
        btn.style.setProperty('padding', '0.35rem 0.75rem')
        btn.style.setProperty('border-radius', '5px')
        btn.style.setProperty('border', '1px solid #d0d7de')
        btn.style.setProperty('font', 'inherit')
        btn.style.setProperty('font-size', '0.85rem')
        btn.style.setProperty('cursor', 'pointer')
        # color variants
        _BTN_COLORS = {
            'primary': ('#0969da', '#fff', '#0969da'),
            'success': ('#1a7f37', '#fff', '#1a7f37'),
            'info':    ('#0550ae', '#fff', '#0550ae'),
            'warning': ('#9a6700', '#fff', '#9a6700'),
            'danger':  ('#cf222e', '#fff', '#cf222e'),
        }
        if self._button_style in _BTN_COLORS:
            bg, fg, bc = _BTN_COLORS[self._button_style]
            btn.style.setProperty('background', bg)
            btn.style.setProperty('color', fg)
            btn.style.setProperty('border-color', bc)

        def _on_click(evt):
            for fn in self._click_callbacks:
                try:
                    fn(self)
                except Exception as e:
                    print(f"[mystral_widgets] on_click error: {e}")

        p = create_proxy(_on_click)
        self._proxies.append(p)
        btn.addEventListener('click', p)
        self._dom = btn


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

class HTML(_Widget):
    def __init__(self, value='', layout=None, **kwargs):
        super().__init__()
        self._value = value
        if layout is not None:
            self._layout = layout

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        if self._dom is not None:
            self._dom.innerHTML = v

    def _build_dom(self):
        div = js.document.createElement('div')
        div.className = 'mw-html'
        div.style.setProperty('font-size', '0.9rem')
        div.innerHTML = self._value
        self._dom = div


# ---------------------------------------------------------------------------
# HTMLMath  (HTML + KaTeX re-render)
# ---------------------------------------------------------------------------

class HTMLMath(_Widget):
    def __init__(self, value='', layout=None, **kwargs):
        super().__init__()
        self._value = value
        if layout is not None:
            self._layout = layout

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        if self._dom is not None:
            self._dom.innerHTML = v
            _render_math(self._dom)

    def _build_dom(self):
        div = js.document.createElement('div')
        div.className = 'mw-htmlmath'
        div.style.setProperty('font-size', '0.9rem')
        div.innerHTML = self._value
        self._dom = div


def _render_math(node):
    """Call KaTeX renderMathInElement if available."""
    try:
        fn = js.globalThis.renderMathInElement
        if fn:
            fn(node)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class Output(_Widget):
    def __init__(self, width='100%', layout=None, **kwargs):
        super().__init__()
        if layout is not None:
            self._layout = layout
        else:
            self._layout = Layout(width=width)

    def _build_dom(self):
        div = js.document.createElement('div')
        div.className = 'mw-output'
        div.style.setProperty('padding', '0.3rem 0')
        self._dom = div

    def clear_output(self):
        if self._dom is not None:
            self._dom.innerHTML = ''

    def __enter__(self):
        self._render()
        _output_stack.append(self)
        return self

    def __exit__(self, *args):
        if _output_stack and _output_stack[-1] is self:
            _output_stack.pop()


# ---------------------------------------------------------------------------
# VBox / HBox
# ---------------------------------------------------------------------------

class VBox(_Widget):
    def __init__(self, children=None, layout=None, **kwargs):
        super().__init__()
        self._children = list(children or [])
        if layout is not None:
            self._layout = layout

    @property
    def children(self):
        return self._children

    def _build_dom(self):
        div = js.document.createElement('div')
        div.className = 'mw-vbox'
        div.style.setProperty('display', 'flex')
        div.style.setProperty('flex-direction', 'column')
        div.style.setProperty('gap', '0.4rem')
        for child in self._children:
            if isinstance(child, _Widget):
                div.appendChild(child._render())
            elif isinstance(child, str):
                span = js.document.createElement('span')
                span.textContent = child
                div.appendChild(span)
        self._dom = div


class HBox(_Widget):
    def __init__(self, children=None, layout=None, **kwargs):
        super().__init__()
        self._children = list(children or [])
        if layout is not None:
            self._layout = layout

    @property
    def children(self):
        return self._children

    def _build_dom(self):
        div = js.document.createElement('div')
        div.className = 'mw-hbox'
        div.style.setProperty('display', 'flex')
        div.style.setProperty('flex-direction', 'row')
        div.style.setProperty('flex-wrap', 'nowrap')
        div.style.setProperty('gap', '0.5rem')
        div.style.setProperty('align-items', 'center')
        div.style.setProperty('overflow-x', 'auto')
        for child in self._children:
            if isinstance(child, _Widget):
                div.appendChild(child._render())
            elif isinstance(child, str):
                span = js.document.createElement('span')
                span.textContent = child
                div.appendChild(span)
        self._dom = div


# ---------------------------------------------------------------------------
# IntText / FloatText / Password
# ---------------------------------------------------------------------------

class IntText(Text):
    def __init__(self, value=0, **kwargs):
        super().__init__(value=str(value), **kwargs)

    @property
    def value(self):
        raw = Text.value.fget(self)
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0

    @value.setter
    def value(self, v):
        Text.value.fset(self, str(v))


class FloatText(Text):
    def __init__(self, value=0.0, **kwargs):
        super().__init__(value=str(value), **kwargs)

    @property
    def value(self):
        raw = Text.value.fget(self)
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0

    @value.setter
    def value(self, v):
        Text.value.fset(self, str(v))


class Password(Text):
    def _build_dom(self):
        super()._build_dom()
        inp = self._dom.querySelector('input')
        if inp:
            inp.type = 'password'


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class Markdown:
    def __init__(self, text=''):
        self.text = text

    def _to_html(self):
        t = self.text
        t = re.sub(r'```([^`]*?)```',
                   lambda m: '<pre><code>' + m.group(1).strip() + '</code></pre>',
                   t, flags=re.DOTALL)
        t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        t = re.sub(r'__(.+?)__', r'<strong>\1</strong>', t)
        t = re.sub(r'\*(.+?)\*', r'<em>\1</em>', t)
        t = re.sub(r'_(.+?)_', r'<em>\1</em>', t)
        t = t.replace('\n', '<br>')
        return t


# ---------------------------------------------------------------------------
# Javascript
# ---------------------------------------------------------------------------

class Javascript:
    def __init__(self, code=''):
        self.code = code


# ---------------------------------------------------------------------------
# display()
# ---------------------------------------------------------------------------

def display(*args, **kwargs):
    out_ctx = _current_output()

    for obj in args:
        if isinstance(obj, Javascript):
            try:
                js.eval(obj.code)
            except Exception as e:
                print(f"[mystral_widgets] Javascript eval error: {e}")
            continue

        if isinstance(obj, Markdown):
            node = js.document.createElement('div')
            node.className = 'mw-markdown'
            node.innerHTML = obj._to_html()
            container = out_ctx._dom if out_ctx is not None else _get_output_container()
            container.appendChild(node)
            continue

        if isinstance(obj, _Widget):
            dom = obj._render()
            container = out_ctx._dom if out_ctx is not None else _get_output_container()
            container.appendChild(dom)
            _render_math(dom)
            continue

        # Fallback
        node = js.document.createElement('pre')
        node.textContent = str(obj)
        container = out_ctx._dom if out_ctx is not None else _get_output_container()
        container.appendChild(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'Layout',
    'Text', 'IntText', 'FloatText', 'Password',
    'Dropdown', 'Checkbox', 'Button',
    'HTML', 'HTMLMath', 'Output', 'VBox', 'HBox',
    'Markdown', 'Javascript', 'display',
]
