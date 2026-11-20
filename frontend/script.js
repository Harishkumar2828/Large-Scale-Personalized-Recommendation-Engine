
const API_URL = "http://127.0.0.1:8000";

const userIdInput = document.getElementById("userId");
const recommendBtn = document.getElementById("recommendBtn");
const message = document.getElementById("message");
const recommendationsContainer = document.getElementById("recommendations");

recommendBtn.addEventListener("click", getRecommendations);

async function getRecommendations() {

    const userId = userIdInput.value.trim();

    if (!userId || !Number.isInteger(Number(userId)) || Number(userId) <= 0) {
        message.textContent = "Please enter a valid User ID.";
        recommendationsContainer.innerHTML = "";
        return;
    }

    recommendBtn.disabled = true;
    recommendBtn.textContent = "Loading...";
    message.textContent = "Finding your personalized movies...";
    recommendationsContainer.innerHTML = "";

    try {

        const response = await fetch(
            `${API_URL}/recommend/${userId}`
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to fetch recommendations.");
        }

        message.textContent =
            `Showing ${data.recommendations_count} recommendations for User ${data.user_id}`;

        displayRecommendations(data.recommendations);

    } catch (error) {

        message.textContent = error.message;
        recommendationsContainer.innerHTML = "";

    } finally {

        recommendBtn.disabled = false;
        recommendBtn.textContent = "Get Recommendations";

    }
}

function displayRecommendations(movies) {

    recommendationsContainer.innerHTML = "";

    if (!movies.length) {
        recommendationsContainer.textContent = "No recommendations found.";
        return;
    }

    movies.forEach(movie => {

        const card = document.createElement("div");
        card.className = "movie-card";

        const title = document.createElement("h3");
        title.textContent = movie.title;

        const movieId = document.createElement("p");
        movieId.className = "movie-id";
        movieId.textContent = `Movie ID: ${movie.movie_id}`;

        const genres = document.createElement("div");

        movie.genres.split("|").forEach(genre => {
            const tag = document.createElement("span");
            tag.className = "genre";
            tag.textContent = genre;
            genres.appendChild(tag);
        });

        const score = document.createElement("p");
        score.className = "score";
        score.textContent = `Recommendation Score: ${movie.score}`;

        card.append(title, movieId, genres, score);

        recommendationsContainer.appendChild(card);

    });
}