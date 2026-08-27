function jsonResponse(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}

function getOrCreateFolder(parent, name) {
  const matches = parent.getFoldersByName(name);
  return matches.hasNext() ? matches.next() : parent.createFolder(name);
}

function replaceFile(folder, filename, blob) {
  const matches = folder.getFilesByName(filename);
  while (matches.hasNext()) {
    matches.next().setTrashed(true);
  }
  return folder.createFile(blob.setName(filename));
}

function doPost(event) {
  try {
    const properties = PropertiesService.getScriptProperties();
    const expectedToken = properties.getProperty("UPLOAD_TOKEN");
    const parentFolderId = properties.getProperty("PARENT_FOLDER_ID");
    if (!expectedToken || !parentFolderId) {
      return jsonResponse({status: "error", message: "Script properties are incomplete"});
    }

    const payload = JSON.parse(event.postData.contents);
    if (!payload.token || payload.token !== expectedToken) {
      return jsonResponse({status: "error", message: "Unauthorized"});
    }
    if (!/^data_\d{1,2}_\d{1,2}$/.test(payload.subfolderName || "")) {
      return jsonResponse({status: "error", message: "Invalid subfolder name"});
    }
    if (!/^[A-Za-z0-9._-]+_master\.csv$/.test(payload.filename || "")) {
      return jsonResponse({status: "error", message: "Invalid filename"});
    }

    const bytes = Utilities.base64Decode(payload.filedata);
    const blob = Utilities.newBlob(bytes, "text/csv", payload.filename);
    const parent = DriveApp.getFolderById(parentFolderId);
    const folder = getOrCreateFolder(parent, payload.subfolderName);
    const file = replaceFile(folder, payload.filename, blob);
    return jsonResponse({status: "success", fileId: file.getId()});
  } catch (error) {
    return jsonResponse({status: "error", message: String(error)});
  }
}
