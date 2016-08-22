(function() {
  'use strict';

  angular
    .module('bubbleApp.models')
    .factory('Tweets', Tweets);

  Tweets.$inject = [
    '$http'
  ];

  function Tweets($http) {
    return {
      get: get
    };

    function get() {
      // hardcoded local folder
      return $http.get('data/tweets.json').then(function(response) {
        return response.data;
      });
    }
  }
})();