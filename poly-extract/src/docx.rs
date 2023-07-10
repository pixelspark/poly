use std::{
	collections::VecDeque,
	io::{Read, Seek},
};

use minidom::Element;
use zip::ZipArchive;

/// Retrieve plain text from a Word DOCX file
pub fn get_text_from_docx<R>(reader: R) -> Option<String>
where
	R: Read + Seek,
{
	let mut result: String = String::new();
	let mut xml_string: String = String::new();

	let mut zip_reader: ZipArchive<R>;
	match ZipArchive::new(reader) {
		Ok(zp) => zip_reader = zp,
		Err(_err) => return None,
	}
	let mut document_xml_file: zip::read::ZipFile<'_>;
	match zip_reader.by_name("word/document.xml") {
		Ok(zpf) => document_xml_file = zpf,
		Err(_err) => return None,
	}

	let _outcome: std::result::Result<usize, std::io::Error> = document_xml_file.read_to_string(&mut xml_string);
	let element: Element = xml_string.parse().unwrap();
	let mut node_que: VecDeque<&Element> = VecDeque::new();
	let mut _text_string: String = String::new();
	node_que.push_back(&element);

	while let Some(node) = node_que.pop_front() {
		if node.name() == "t" {
			result.push_str(&node.text());
			result.push('\n');
		}
		for child in node.children() {
			node_que.push_back(child);
		}
	}
	if result.is_empty() {
		result.push_str("   ");
	}
	Some(result)
}
