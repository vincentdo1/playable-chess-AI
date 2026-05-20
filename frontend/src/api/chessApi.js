export class ChessApi {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
  }

  async health() {
    const response = await fetch(`${this.apiUrl}/`);
    return response.json();
  }

  async move(payload) {
    const response = await fetch(`${this.apiUrl}/api/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return response.json();
  }
}
