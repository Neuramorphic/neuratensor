# Cómo habilitar Trusted Publishing para NeuraTensor

## Requisitos previos

- ✅ Ya tienes el paquete `neuratensor` publicado en PyPI
- ✅ Tienes acceso de "Owner" o "Maintainer" al proyecto en PyPI
- ⚠️ **NECESITAS:** Un repositorio de GitHub con el código de neuratensor

## ¿Tienes un repo de GitHub?

**SI NO TIENES REPO de GitHub aún**, primero necesitas:

```bash
# 1. Crea un repo en GitHub (https://github.com/new)
#    Nombre: neuratensor
#    Owner: neuramorphic (o tu usuario/org)
#    Público o privado (ambos funcionan)

# 2. Sube el código
cd /Users/peterfulle/neuratensor-v3
git init
git add .
git commit -m "Initial commit - NeuraTensor v4.0.3"
git branch -M main
git remote add origin https://github.com/TUUSUARIO/neuratensor.git
git push -u origin main

# 3. Sube el workflow
git add .github/workflows/publish.yml
git commit -m "Add Trusted Publishing workflow"
git push
```

**SI YA TIENES REPO**, continúa abajo.

---

## Paso a paso: Configurar Trusted Publishing en PyPI

### 1. Entra a PyPI y ve al proyecto

1. Abre: https://pypi.org/manage/account/
2. Busca "neuratensor" en la lista "Your projects"
3. Haz clic en **"neuratensor"**

### 2. Ve a la sección Publishing

En la página del proyecto, verás un menú lateral izquierdo:
- Project description
- Releases  
- **Publishing** ← **HAZ CLIC AQUÍ**
- Collaboration
- Settings

### 3. Agregar Trusted Publisher

En la sección "Publishing", verás:

```
Publishing

This project is not currently using any trusted publishers.

[Add a new pending publisher]  ← **HAZ CLIC EN ESTE BOTÓN**
```

### 4. Seleccionar GitHub

Te preguntará "Select provider":
- GitHub ← **SELECCIONA ESTE**
- GitLab
- Google

### 5. Llenar el formulario

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Add a new pending publisher for neuratensor

Provider: GitHub Actions

Owner:               [________________]  ← neuramorphic (o tu usuario)
Repository name:     [________________]  ← neuratensor
Workflow name:       [________________]  ← publish.yml
Environment name:    [________________]  ← (DEJAR VACÍO o poner "release")

[Add]  [Cancel]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**IMPORTANTE:** El "Owner" debe coincidir EXACTAMENTE con el owner del repo de GitHub.

Por ejemplo:
- Si tu repo es `https://github.com/neuramorphic/neuratensor` → Owner = `neuramorphic`
- Si tu repo es `https://github.com/peterfulle/neuratensor` → Owner = `peterfulle`

### 6. Guardar

Haz clic en **"Add"**

Verás:

```
✓ Pending publisher added successfully

Trusted publishers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GitHub: neuramorphic/neuratensor @ publish.yml

[Remove]
```

---

## ¿Cómo funciona ahora?

### Primera publicación desde GitHub

1. Ve a tu repo de GitHub
2. Crea un **Release** (no solo un tag):
   - Releases → "Create a new release"
   - Tag: `v4.1.0`
   - Title: `NeuraTensor v4.1.0`
   - Description: "First release with Trusted Publishing"
   - Haz clic en "Publish release"

3. GitHub Actions ejecutará automáticamente el workflow `publish.yml`
4. El workflow:
   - Construye el paquete
   - Se autentica con PyPI usando OIDC (sin token)
   - Publica el paquete
   - PyPI genera **provenance attestation** automáticamente

### Verificar que funcionó

1. Ve a https://pypi.org/project/neuratensor/4.1.0/
2. Deberías ver un badge: **"✓ Verified by GitHub"**
3. Haz clic en "View attestations" para ver la firma criptográfica

---

## Troubleshooting

### Error: "Invalid or missing id token"

**Problema:** El workflow no tiene permisos de `id-token: write`

**Solución:** Verifica que `.github/workflows/publish.yml` tiene:

```yaml
permissions:
  contents: read
  id-token: write  # <-- ESTO ES CRÍTICO
```

### Error: "Trusted publisher mismatch"

**Problema:** El owner/repository en PyPI no coincide con GitHub

**Solución:** Verifica que:
- PyPI config: `Owner = neuramorphic`, `Repository = neuratensor`
- GitHub repo: `https://github.com/neuramorphic/neuratensor`

Los nombres deben ser **exactamente iguales** (case-sensitive).

### Error: "Workflow not found"

**Problema:** El archivo `publish.yml` no existe en `.github/workflows/`

**Solución:**

```bash
cd /Users/peterfulle/neuratensor-v3
mkdir -p .github/workflows
# El archivo ya debería existir, pero si no:
ls -la .github/workflows/publish.yml

# Si no existe, vuelve a crearlo
```

---

## Ejemplo de release manual para testing

Puedes probar el workflow sin hacer un release:

1. Ve a tu repo en GitHub
2. Actions → "Publish to PyPI"
3. Click "Run workflow" → selecciona branch "main"
4. Click "Run workflow"

Esto ejecutará el workflow manualmente (gracias a `workflow_dispatch` en el YAML).

---

## Notas importantes

1. **No borres el API token todavía** — déjalo hasta confirmar que el Trusted Publishing funciona
2. **Primera publicación con Trusted Publishing debe ser una versión nueva** (no puedes re-publicar v4.0.3)
3. **El workflow solo se ejecuta en releases** o manual dispatch (no en cada push)
4. **Los pull requests NO publican a PyPI** (por seguridad)

---

## ¿Preguntas?

Si algo no funciona, revisa los logs del workflow:
- GitHub repo → Actions → Click en el run fallido
- Lee el error en el paso "Publish to PyPI"

Los errores más comunes están documentados arriba en "Troubleshooting".
