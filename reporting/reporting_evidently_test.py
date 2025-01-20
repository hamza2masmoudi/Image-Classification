import evidently
import pkgutil

print("Modules in Evidently:")
for module in pkgutil.iter_modules(evidently.__path__):
    print(module.name)