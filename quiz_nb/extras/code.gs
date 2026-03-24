
const SECRET = "CLE_DE_LECTURE_A_MODIFIER!";  // à garder privé // utilisé pour lire les résultats complets
const KEY = "IOK33";  // à garder privé // utilisé pour tester la connexion internet (le sheet répond)
const SHEET_NAME = "Feuille 1";           // nom de l'onglet
const CONFIG_SHEET = "Config";           // nom de l'onglet
const SPREADSHEET_ID = "0"; //"1KXxF98JuhlaRKap9484o0zpdEhqs0PkTQsmREcKEsSM";     // optionnel si lié au sheet

function doPost(e) {

  if (!canReceiveData()) {
    return; // ou return unauthorized(); // on ignore tout simplement les données
  }

  const config = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CONFIG_SHEET);

  // NKEEP : On ne garde que les NKEEP derniers enregistrements
  // NMAX : Nettoyage quand on atteint NMAX enregistrements
  const NMAX = config.getRange("A5").getValue() || 5000 //valeur par défaut: 5000
  const NKEEP = config.getRange("B5").getValue() || 6000

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);

  try {
    // 1. Extraction des données
    const data = JSON.parse(e.postData.contents);

    // 2. Écriture dans la feuille
    sheet.appendRow([
      new Date(),
      data.timestamp || "",
      data.notebook_id || "",
      data.student || "",
      data.quiz_title || "",
      data.event_type || "",
      // Si parameters est un objet ou une liste, on le transforme en texte pour la cellule
      typeof data.parameters === 'object' ? JSON.stringify(data.parameters) : (data.parameters || ""),
      // Si answers est un objet ou une liste, on le transforme en texte pour la cellule
      typeof data.answers === 'object' ? JSON.stringify(data.answers) : (data.answers || ""),
      data.score || ""
    ]);

    // On supprime les données pour éviter que la feuille ne se remplisse à l'infini

    if (sheet.getLastRow() > NMAX) {
        enforceMaxRows(sheet, NKEEP);
      }


    // 3. Réponse (on utilise TEXT pour maximiser la compatibilité CORS)
    return ContentService
      .createTextOutput(JSON.stringify({ "status": "ok" }))
      .setMimeType(ContentService.MimeType.TEXT);

  } catch (err) {
    // En cas d'erreur, on renvoie le message en texte brut
    return ContentService
      .createTextOutput(JSON.stringify({ "status": "error", "message": err.toString() }))
      .setMimeType(ContentService.MimeType.TEXT);
  }
}

function doGet(e) {

  const key = e.parameter.key;

  if (!e.parameter || e.parameter.key === KEY) {
    return internetOK(e);

  } else if (e.parameter.secret === SECRET) {
    return sendData(e);

  } else {
    return unauthorized();
  }
}

// ------------------------------------------

function canReceiveData() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet()
    .getSheetByName(CONFIG_SHEET);

  return sheet.getRange("B2").getValue() === true;
}

function enforceMaxRows(sheet, maxRows) {
  const lastRow = sheet.getLastRow();
  const headerRows = 1; // nombre de lignes d'en-tête

  const excess = lastRow - headerRows - maxRows;

  if (excess > 0) {
    sheet.deleteRows(headerRows + 1, excess);
  }
}


function internetOK(e) {
  const cell = e.parameter.cell || "A2";
  const sheet = SpreadsheetApp
    .getActiveSpreadsheet()
    .getSheetByName(CONFIG_SHEET);

  const value = sheet.getRange(cell).getValue();

  return ContentService
    .createTextOutput(JSON.stringify({ value: value }))
    .setMimeType(ContentService.MimeType.JSON);
}



function unauthorized() {
    return ContentService
      .createTextOutput("Unauthorized")
      .setMimeType(ContentService.MimeType.TEXT);
  }

function sendData(e) {
    //const ss = SPREADSHEET_ID
    //    ? SpreadsheetApp.openById(SPREADSHEET_ID)
    //    : SpreadsheetApp.getActiveSpreadsheet();

    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getSheetByName(SHEET_NAME);
    if (!sheet) {
        return ContentService
        .createTextOutput("Sheet not found")
        .setMimeType(ContentService.MimeType.TEXT);
    }

    // --- Lecture des données ---
    const data = sheet.getDataRange().getValues();

    // --- Conversion en CSV ---
    const csv = data.map(row =>
        row.map(cell =>
        `"${String(cell).replace(/"/g, '""')}"`
        ).join(",")
    ).join("\n");

    return ContentService
        .createTextOutput(csv)
        .setMimeType(ContentService.MimeType.CSV);
}
