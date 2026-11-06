"""Temporarily freeze selected IDs without retaining request loader cycles."""
from contextlib import contextmanager

@contextmanager
def fixed_selection(loader, selected):
    name = "configure_impress_layer"
    had_override = name in vars(loader)
    previous_override = vars(loader).get(name)
    configure = loader.configure_impress_layer
    def fixed_configure(*, layer, selected_tokens, prefetch_priority_tokens=None):
        ids = selected[layer]
        return configure(layer=layer,selected_tokens=ids,prefetch_priority_tokens=ids)
    setattr(loader,name,fixed_configure)
    try:
        yield
    finally:
        # Restoring a newly bound class method into the instance dict creates
        # loader -> bound_method -> loader. Restore attribute ownership instead.
        if had_override:
            setattr(loader,name,previous_override)
        else:
            delattr(loader,name)
